//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type apiMediaProvider struct {
	*embedding.FuncProvider
	imageError error
	audio      *embedding.AudioRequest
}

func (p *apiMediaProvider) EmbedImage(context.Context, []byte, int) ([]float32, error) {
	return []float32{1, 0}, p.imageError
}

func (p *apiMediaProvider) EmbedAudio(_ context.Context, request embedding.AudioRequest) ([]float32, error) {
	p.audio = &request
	return []float32{0, 1}, nil
}

func (p *apiMediaProvider) EmbeddingInfo() embedding.ModelInfo {
	return embedding.ModelInfo{Dimension: 2, Dimensions: []int{2}, Modalities: []string{"text", "image", "audio"}, Audio: &binding.AudioCapability{MaxSeconds: 30, MaxChannels: 8}}
}

func apiAudioFixture() string {
	wav := make([]byte, 48)
	copy(wav, "RIFF")
	binary.LittleEndian.PutUint32(wav[4:], 40)
	copy(wav[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(wav[16:], 16)
	binary.LittleEndian.PutUint16(wav[20:], 1)
	binary.LittleEndian.PutUint16(wav[22:], 2)
	binary.LittleEndian.PutUint32(wav[24:], 44100)
	binary.LittleEndian.PutUint32(wav[28:], 176400)
	binary.LittleEndian.PutUint16(wav[32:], 4)
	binary.LittleEndian.PutUint16(wav[34:], 16)
	copy(wav[36:], "data")
	binary.LittleEndian.PutUint32(wav[40:], 4)
	binary.LittleEndian.PutUint16(wav[44:], 16384)
	binary.LittleEndian.PutUint16(wav[46:], 49152)
	return "data:audio/wav;base64," + base64.StdEncoding.EncodeToString(wav)
}

func TestOwnedEmbeddingMediaUsesSelectedModelAndOriginalPCM(t *testing.T) {
	fp, _ := embedding.NewFuncProvider("synthetic", 2, func(context.Context, string) ([]float32, error) { return []float32{1, 0}, nil })
	chosen := &apiMediaProvider{FuncProvider: fp}
	wrong := &apiMediaProvider{FuncProvider: fp, imageError: errors.New("wrong provider selected")}
	prepared := embedding.NewSet(map[string]embedding.Provider{"multimodal": wrong, "chosen": chosen}, "multimodal")
	request := EmbeddingRequest{Model: "chosen", Texts: []string{"text"}, Images: []string{"data:image/png;base64,YQ=="}, Audios: []string{apiAudioFixture()}}
	applyEmbeddingDefaults(&request)
	if request.Dimension != 0 {
		t.Fatalf("default overwrote model dimension: %+v", request)
	}
	if _, _, valid := validateEmbeddingRequest(request, nil); !valid {
		t.Fatal("rejected typed media request")
	}
	results, _, err := buildOwnedEmbeddingResults(context.Background(), prepared, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 || results[1].Modality != "image" || results[2].Modality != "audio" {
		t.Fatalf("missing modality results: %+v", results)
	}
	for _, result := range results {
		if result.ModelUsed != "chosen" || result.Dimension != 2 {
			t.Fatalf("wrong representation: %+v", result)
		}
	}
	if chosen.audio == nil || chosen.audio.SampleRate != 44100 || chosen.audio.Channels != 2 || chosen.audio.PCM[0] != .5 || chosen.audio.PCM[1] != -.5 {
		t.Fatalf("original audio was changed: %+v", chosen.audio)
	}
	request.Dimension = 1
	if _, _, err := buildOwnedEmbeddingResults(context.Background(), prepared, request); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("unsupported dimension not rejected: %v", err)
	}
}

func TestEmbeddingMediaDefaultsSelectOneVectorSpace(t *testing.T) {
	request := EmbeddingRequest{Texts: []string{"anchor"}, Audios: []string{apiAudioFixture()}}
	applyEmbeddingDefaults(&request)
	fp, _ := embedding.NewFuncProvider("synthetic", 2, func(context.Context, string) ([]float32, error) { return []float32{1, 0}, nil })
	provider := &apiMediaProvider{FuncProvider: fp}
	prepared := embedding.NewSet(map[string]embedding.Provider{"recipe-encoder": provider}, "recipe-encoder")
	results, _, selectErr := buildOwnedEmbeddingResults(context.Background(), prepared, request)
	if selectErr != nil || len(results) != 2 {
		t.Fatalf("auto media failed for prepared alias: %v", selectErr)
	}
	for _, result := range results {
		if result.ModelUsed != "recipe-encoder" {
			t.Fatalf("mixed modalities crossed representations: %+v", result)
		}
	}
	for _, err := range []error{errors.New("execution failed"), binding.ErrClosed, binding.ErrInvalidResult} {
		status, _, _ := classifyEmbeddingError(&mediaEncodeError{modality: "audio", index: 0, err: err})
		if status != http.StatusInternalServerError {
			t.Fatalf("provider failure misclassified as bad audio: %v -> %d", err, status)
		}
	}
}

func TestEmbeddingAutoMediaRequiresActualCapabilities(t *testing.T) {
	fp, _ := embedding.NewFuncProvider("synthetic", 2, func(context.Context, string) ([]float32, error) { return []float32{1, 0}, nil })
	provider := &apiMediaProvider{FuncProvider: fp}
	for _, prepared := range []*embedding.Set{
		embedding.NewSet(map[string]embedding.Provider{"multimodal": fp}, "multimodal"),
		embedding.NewSet(map[string]embedding.Provider{"chosen": provider}, "chosen"),
	} {
		dimension := 0
		if prepared.Has("chosen") {
			dimension = 3
		}
		_, _, err := buildOwnedEmbeddingResults(context.Background(), prepared, EmbeddingRequest{Audios: []string{apiAudioFixture()}, Dimension: dimension})
		if !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("unsupported media selection accepted: %v", err)
		}
	}
}

func TestOwnedImageCanonicalizesAcceptedDataURI(t *testing.T) {
	fp, _ := embedding.NewFuncProvider("synthetic", 2, func(context.Context, string) ([]float32, error) { return []float32{1, 0}, nil })
	provider := &apiMediaProvider{FuncProvider: fp}
	prepared := embedding.NewSet(map[string]embedding.Provider{"vision": provider}, "vision")
	_, _, err := buildOwnedEmbeddingResults(context.Background(), prepared, EmbeddingRequest{Images: []string{"DATA:IMAGE/PNG;BASE64,YQ=="}})
	if err != nil {
		t.Fatalf("validated uppercase data URI was not canonicalized: %v", err)
	}
}

func TestEmbeddingAPIAutoMediaExplicitRecipeHTTP(t *testing.T) {
	fp, _ := embedding.NewFuncProvider("synthetic", 2, func(context.Context, string) ([]float32, error) { return []float32{1, 0}, nil })
	provider := &apiMediaProvider{FuncProvider: fp}
	prepared := embedding.NewSet(map[string]embedding.Provider{"mmbert": provider}, "mmbert")
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"vision": {Provider: "ort", Device: "cpu", Artifact: t.TempDir()}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "vision", Contract: "embedding.v1", Adapter: "vela_omni"}}
	// The generation owns an already prepared encoder; the API must select its
	// actual capabilities without interpreting the legacy catalog alias.
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil, classification.RecipeRuntimeOptions{Embeddings: prepared})
	if err != nil {
		t.Fatal(err)
	}
	service := services.NewRecipeClassificationService(classifiers, cfg)
	t.Cleanup(func() { _ = service.Close() })
	api := &ClassificationAPIServer{config: cfg, classificationSvc: service}
	body, err := json.Marshal(EmbeddingRequest{Recipe: "default", Texts: []string{"anchor"}, Audios: []string{apiAudioFixture()}, Images: []string{"data:image/png;base64,YQ=="}})
	if err != nil {
		t.Fatal(err)
	}
	recorder := httptest.NewRecorder()
	api.handleEmbeddings(recorder, httptest.NewRequest(http.MethodPost, "/api/v1/embeddings", strings.NewReader(string(body))))
	if recorder.Code != http.StatusOK {
		t.Fatalf("HTTP %d: %s", recorder.Code, recorder.Body.String())
	}
	var response EmbeddingResponse
	if err = json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Recipe != "default" || len(response.Embeddings) != 3 {
		t.Fatalf("wrong recipe or results: %+v", response)
	}
	for _, result := range response.Embeddings {
		if result.ModelUsed != "mmbert" || result.Dimension != 2 {
			t.Fatalf("lost scoped representation: %+v", result)
		}
	}
}
