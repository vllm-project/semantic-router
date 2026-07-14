//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestEmbeddingProcessAdmissionCapacityReleaseAndCancellation(t *testing.T) {
	admission := newEmbeddingProcessAdmission(embeddingProcessAdmissionCapacity)
	releaseFirst, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("acquire first slot: %v", err)
	}
	releaseSecond, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("acquire second slot: %v", err)
	}
	if _, overloadErr := admission.TryAcquire(context.Background()); !errors.Is(overloadErr, errEmbeddingOverloaded) {
		t.Fatalf("expected fail-fast overload, got %v", overloadErr)
	}

	releaseFirst()
	releaseFirst()
	reacquired, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("reacquire released slot: %v", err)
	}
	reacquired()
	releaseSecond()

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := admission.TryAcquire(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context cancellation, got %v", err)
	}
}

func TestAPIServerFallbackUsesSharedProcessAdmission(t *testing.T) {
	server := &ClassificationAPIServer{}
	if got := server.embeddingProcessAdmission(); got != embedding.DefaultProcessAdmission {
		t.Fatalf("API server fallback admission = %p, want shared process admission %p", got, embedding.DefaultProcessAdmission)
	}
}

func TestEmbeddingProcessAdmissionReleaseIsRaceSafe(t *testing.T) {
	admission := newEmbeddingProcessAdmission(1)
	release, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("acquire slot: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			release()
		}()
	}
	wg.Wait()

	reacquired, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("slot leaked after concurrent release: %v", err)
	}
	reacquired()
}

func TestEmbeddingHandlersReturnStableOverloadContract(t *testing.T) {
	counters := installEmbeddingWorkCounters(t)

	admission := newEmbeddingProcessAdmission(embeddingProcessAdmissionCapacity)
	releaseFirst, _ := admission.TryAcquire(context.Background())
	releaseSecond, _ := admission.TryAcquire(context.Background())
	defer releaseFirst()
	defer releaseSecond()

	imageURI := mustEmbeddingImageDataURI(t, "image/png")
	imageBody, err := json.Marshal(map[string]interface{}{"images": []string{imageURI}})
	if err != nil {
		t.Fatalf("marshal image request: %v", err)
	}
	server := &ClassificationAPIServer{
		classificationSvc:  services.NewPlaceholderClassificationService(),
		embeddingAdmission: admission,
	}
	tests := []struct {
		name    string
		path    string
		body    string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{name: "embedding text", path: "/api/v1/embeddings", body: `{"texts":["hello"]}`, handler: server.handleEmbeddings},
		{name: "embedding image", path: "/api/v1/embeddings", body: string(imageBody), handler: server.handleEmbeddings},
		{name: "similarity", path: "/api/v1/similarity", body: `{"text1":"hello","text2":"world"}`, handler: server.handleSimilarity},
		{name: "batch similarity", path: "/api/v1/similarity/batch", body: `{"query":"hello","candidates":["world"]}`, handler: server.handleBatchSimilarity},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()

			tt.handler(rr, req)

			if rr.Code != http.StatusServiceUnavailable {
				t.Fatalf("expected 503, got %d: %s", rr.Code, rr.Body.String())
			}
			assertJSONErrorCode(t, rr.Body.Bytes(), "EMBEDDING_OVERLOADED")
			assertJSONErrorMessage(t, rr.Body.Bytes(), "embedding inference is temporarily overloaded")
			if rr.Header().Get("Retry-After") != "1" {
				t.Fatalf("missing Retry-After contract: %v", rr.Header())
			}
			if rr.Header().Get("Cache-Control") != "no-store" {
				t.Fatalf("missing no-store contract: %v", rr.Header())
			}
		})
	}
	if counters.validator.Load() != 0 || counters.native.Load() != 0 {
		t.Fatalf("overloaded handlers performed expensive work: validator=%d native=%d", counters.validator.Load(), counters.native.Load())
	}
}

func TestIntentClassificationRemoteConfigWithUnresolvedLocalOverrideFailsBuild(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendCandle)
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{
			EmbeddingConfig: config.HNSWConfig{
				Backend:   config.EmbeddingBackendOpenAICompatible,
				ModelType: config.EmbeddingModelTypeRemote,
			},
			Endpoint: config.EmbeddingEndpointConfig{
				BaseURL: "https://embedding.invalid/v1",
				Model:   "remote-embedding",
			},
		}},
		IntelligentRouting: config.IntelligentRouting{Signals: config.Signals{
			EmbeddingRules: []config.EmbeddingRule{{
				Name:       "remote-config-local-override",
				Candidates: []string{"billing"},
			}},
		}},
	}
	if classifier, err := classification.BuildClassifier(cfg, nil, nil, nil); classifier != nil || err == nil {
		t.Fatalf("unresolved local override = classifier %v err %v, want fail-closed build", classifier, err)
	}
}

type embeddingWorkCounters struct {
	validator atomic.Int32
	native    atomic.Int32
}

func installEmbeddingWorkCounters(t *testing.T) *embeddingWorkCounters {
	t.Helper()
	originalValidator := validateImagesAfterAdmission
	originalText := embeddingOutputForRequest
	originalImage := multiModalEncodeImage
	originalSimilarity := calculateEmbeddingSimilarityNative
	originalBatch := calculateSimilarityBatchNative
	counters := &embeddingWorkCounters{}
	validateImagesAfterAdmission = func([]string) (string, string, bool) {
		counters.validator.Add(1)
		return "", "", true
	}
	embeddingOutputForRequest = func(EmbeddingRequest, string) (*candle_binding.EmbeddingOutput, error) {
		counters.native.Add(1)
		return nil, errors.New("must not run")
	}
	multiModalEncodeImage = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		counters.native.Add(1)
		return nil, errors.New("must not run")
	}
	calculateEmbeddingSimilarityNative = func(string, string, candle_binding.SimilarityOptions) (*candle_binding.SimilarityOutput, error) {
		counters.native.Add(1)
		return nil, errors.New("must not run")
	}
	calculateSimilarityBatchNative = func(string, []string, int, candle_binding.SimilarityOptions) (*candle_binding.BatchSimilarityOutput, error) {
		counters.native.Add(1)
		return nil, errors.New("must not run")
	}
	t.Cleanup(func() {
		validateImagesAfterAdmission = originalValidator
		embeddingOutputForRequest = originalText
		multiModalEncodeImage = originalImage
		calculateEmbeddingSimilarityNative = originalSimilarity
		calculateSimilarityBatchNative = originalBatch
	})
	return counters
}

func TestEmbeddingHandlerReleasesAdmissionOnNativeError(t *testing.T) {
	original := embeddingOutputForRequest
	embeddingOutputForRequest = func(EmbeddingRequest, string) (*candle_binding.EmbeddingOutput, error) {
		return nil, errors.New("private native failure")
	}
	t.Cleanup(func() { embeddingOutputForRequest = original })

	admission := newEmbeddingProcessAdmission(1)
	server := &ClassificationAPIServer{embeddingAdmission: admission}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings", strings.NewReader(`{"texts":["hello"]}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	server.handleEmbeddings(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected native error status 500, got %d: %s", rr.Code, rr.Body.String())
	}
	release, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("native error leaked admission slot: %v", err)
	}
	release()
}
