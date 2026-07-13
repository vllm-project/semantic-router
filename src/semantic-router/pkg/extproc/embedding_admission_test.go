package extproc

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func localSelectionAdmissionRouter(admission *embedding.ProcessAdmission) *OpenAIRouter {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				Qwen3ModelPath: "models/test-qwen3",
				EmbeddingConfig: config.HNSWConfig{
					ModelType: config.EmbeddingModelTypeQwen3,
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:      "local-selection",
				ModelRefs: []config.ModelRef{{Model: "small"}, {Model: "large"}},
				Algorithm: &config.AlgorithmConfig{Type: "router_dc"},
			}},
		},
	}
	return &OpenAIRouter{
		Config:             cfg,
		Classifier:         &classification.Classifier{Config: cfg},
		embeddingAdmission: admission,
	}
}

func TestExtProcFallbackUsesSharedProcessAdmission(t *testing.T) {
	router := &OpenAIRouter{}
	if got := router.embeddingProcessAdmission(); got != embedding.DefaultProcessAdmission {
		t.Fatalf("ExtProc fallback admission = %p, want shared process admission %p", got, embedding.DefaultProcessAdmission)
	}
}

func TestDecisionAdmissionFailsFastForTextAndImageAndReleases(t *testing.T) {
	admission := embedding.NewProcessAdmission(1)
	heldRelease, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("occupy admission: %v", err)
	}

	router := localSelectionAdmissionRouter(admission)
	if _, imageErr := router.admitDecisionEvaluation(context.Background(), "MoM", "data:image/png;base64,AA=="); !errors.Is(imageErr, embedding.ErrOverloaded) {
		t.Fatalf("image decision admission error = %v, want overload", imageErr)
	}
	if _, textErr := router.admitDecisionEvaluation(context.Background(), "MoM", ""); !errors.Is(textErr, embedding.ErrOverloaded) {
		t.Fatalf("text decision admission error = %v, want overload", textErr)
	}

	heldRelease()
	imageRelease, err := router.admitDecisionEvaluation(context.Background(), "MoM", "data:image/png;base64,AA==")
	if err != nil {
		t.Fatalf("acquire released image slot: %v", err)
	}
	imageRelease()
}

func TestDecisionAdmissionSkipsPureStaticConfiguration(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	release, err := router.admitDecisionEvaluation(context.Background(), "explicit-model", "")
	if err != nil {
		t.Fatalf("pure static admission returned error: %v", err)
	}
	release()
}

func TestDecisionAdmissionDoesNotBypassInvalidLocalRuntimePlan(t *testing.T) {
	admission := embedding.NewProcessAdmission(1)
	heldRelease, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("occupy admission: %v", err)
	}
	defer heldRelease()

	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{{
					Name:      "invalid-local-selection",
					ModelRefs: []config.ModelRef{{Model: "small"}, {Model: "large"}},
					Algorithm: &config.AlgorithmConfig{Type: "router_dc"},
				}},
			},
		},
		embeddingAdmission: admission,
	}

	_, err = router.admitDecisionEvaluation(context.Background(), "MoM", "")
	if !errors.Is(err, embedding.ErrOverloaded) {
		t.Fatalf("invalid local runtime admission error = %v, want overload", err)
	}
}

func TestExtProcRemoteConfigWithAmbiguousLocalOverrideFailsBuild(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendOpenVINO)
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
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{EmbeddingRules: []config.EmbeddingRule{{
				Name:       "remote-config-local-override",
				Candidates: []string{"billing"},
			}}},
			Decisions: []config.Decision{{
				Name:      "billing-route",
				Rules:     config.RuleCombination{Type: config.SignalTypeEmbedding, Name: "remote-config-local-override"},
				ModelRefs: []config.ModelRef{{Model: "default-model"}},
			}},
		},
		BackendModels: config.BackendModels{DefaultModel: "default-model"},
	}
	if _, err := classification.BuildClassifier(cfg, nil, nil, nil); err == nil || !strings.Contains(err.Error(), "requires an explicit local model") {
		t.Fatalf("local override build error = %v", err)
	}
}

func TestClassificationOverloadResponseIsStableAndNonCacheable(t *testing.T) {
	router := &OpenAIRouter{}
	response := router.classificationEvaluationErrorResponse(embedding.ErrOverloaded)
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("expected immediate overload response")
	}
	if got := immediate.GetStatus().GetCode(); got != typev3.StatusCode_ServiceUnavailable {
		t.Fatalf("status = %v, want service unavailable", got)
	}
	headers := headerValuesByName(immediate.GetHeaders().GetSetHeaders())
	if got := headers["cache-control"]; got != "no-store" {
		t.Fatalf("cache-control = %q, want no-store", got)
	}
	if got := headers["retry-after"]; got != "1" {
		t.Fatalf("retry-after = %q, want 1", got)
	}
	if body := string(immediate.GetBody()); !containsAll(body, embeddingOverloadMessage, `"code":503`) {
		t.Fatalf("overload body = %q", body)
	}
}

func TestImageEvaluationErrorsAreContentSafeAndNonCacheable(t *testing.T) {
	router := &OpenAIRouter{}
	tests := []struct {
		name       string
		err        error
		wantStatus typev3.StatusCode
		wantBody   string
	}{
		{
			name:       "invalid input",
			err:        errors.Join(classification.ErrImageSignalEvaluation, classification.ErrInvalidImageSignalInput),
			wantStatus: typev3.StatusCode_BadRequest,
			wantBody:   "image input must contain a decodable JPEG or PNG image within the supported limits",
		},
		{
			name:       "internal failure",
			err:        fmt.Errorf("%w: private model path", classification.ErrImageSignalEvaluation),
			wantStatus: typev3.StatusCode_InternalServerError,
			wantBody:   "router image evaluation failed",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := router.classificationEvaluationErrorResponse(tt.err)
			immediate := response.GetImmediateResponse()
			if got := immediate.GetStatus().GetCode(); got != tt.wantStatus {
				t.Fatalf("status = %v, want %v", got, tt.wantStatus)
			}
			if body := string(immediate.GetBody()); !strings.Contains(body, tt.wantBody) || strings.Contains(body, "private model path") {
				t.Fatalf("unsafe image evaluation body = %q", body)
			}
			if got := headerValuesByName(immediate.GetHeaders().GetSetHeaders())["cache-control"]; got != "no-store" {
				t.Fatalf("cache-control = %q, want no-store", got)
			}
		})
	}
}

func TestTextEvaluationErrorIsContentSafeAndNonCacheable(t *testing.T) {
	router := &OpenAIRouter{}
	response := router.classificationEvaluationErrorResponse(
		fmt.Errorf("%w: private provider detail", classification.ErrTextSignalEvaluation),
	)
	immediate := response.GetImmediateResponse()
	if got := immediate.GetStatus().GetCode(); got != typev3.StatusCode_ServiceUnavailable {
		t.Fatalf("status = %v, want service unavailable", got)
	}
	body := string(immediate.GetBody())
	if !strings.Contains(body, "router signal evaluation temporarily unavailable") || strings.Contains(body, "private provider detail") {
		t.Fatalf("unsafe text evaluation body = %q", body)
	}
	if got := headerValuesByName(immediate.GetHeaders().GetSetHeaders())["cache-control"]; got != "no-store" {
		t.Fatalf("cache-control = %q, want no-store", got)
	}
	if got := headerValuesByName(immediate.GetHeaders().GetSetHeaders())["retry-after"]; got != "1" {
		t.Fatalf("retry-after = %q, want 1", got)
	}
}

func TestUnknownEvaluationErrorDoesNotLeak(t *testing.T) {
	response := (&OpenAIRouter{}).classificationEvaluationErrorResponse(errors.New("private evaluator detail"))
	immediate := response.GetImmediateResponse()
	if immediate.GetStatus().GetCode() != typev3.StatusCode_InternalServerError || strings.Contains(string(immediate.GetBody()), "private evaluator detail") {
		t.Fatalf("unknown evaluation response = %+v", immediate)
	}
	if headerValuesByName(immediate.GetHeaders().GetSetHeaders())["cache-control"] != "no-store" {
		t.Fatal("unknown evaluation error was cacheable")
	}
}

func containsAll(value string, fragments ...string) bool {
	for _, fragment := range fragments {
		if !strings.Contains(value, fragment) {
			return false
		}
	}
	return true
}
