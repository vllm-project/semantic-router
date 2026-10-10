//go:build !windows

package apiserver

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

func TestTypedClassificationEnvelopeValidation(t *testing.T) {
	api := newBatchClassificationTestServer(batchClassificationMaxBatchSizeConfig(2))
	for _, body := range []string{
		`{}`, `null`, `{"inputs":null}`, `{"inputs":[]}`, `{"inputs":{}}`,
		`{"texts":null,"inputs":[]}`, `{"texts":[],"inputs":[{"type":"text","text":"x"}]}`,
		`{"texts":["x"],"inputs":null}`, `{"inputs":[{}, {}, {}]}`,
		`{"inputs":[{}],"Texts":null}`, `{"Inputs":[{}],"texts":[]}`, `{"inputs":[{}],"textſ":null}`,
		`{"inputs":[{}],"inputs":[{}]}`, `{"inputs":[{}],"INPUTS":[{}]}`,
		`{"inputs":[{}],"task_type":"audio"}`, `{"inputs":[{}]} {}`,
	} {
		t.Run(body, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			api.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("got %d: %s", recorder.Code, recorder.Body.String())
			}
		})
	}
	for _, body := range []string{
		`{"input":null}`, `{"INPUT":null}`,
		`{"input":{"type":"text","text":"x"},"Text":""}`,
		`{"INPUT":{"type":"text","text":"x"},"Messages":null}`,
		`{"input":{"type":"text","text":"x"},"messageſ":[]}`,
		`{"input":{"type":"text","text":"x"},"INPUT":null}`, `{"input":{"type":"text","text":"x"},"text":""}`,
		`{"input":{"type":"text","text":"x"},"messages":null}`,
		`{"input":{"type":"text","text":"x"},"messages":[]}`,
	} {
		t.Run(body, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			api.handleIntentClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("got %d: %s", recorder.Code, recorder.Body.String())
			}
		})
	}
	// Each malformed typed item is an item error, rather than an envelope error.
	recorder := httptest.NewRecorder()
	api.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"inputs":[null,42]}`)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("got %d: %s", recorder.Code, recorder.Body.String())
	}
	var response BatchClassificationResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	for index, result := range response.Results {
		if result.Index == nil || *result.Index != index || result.Error == nil || result.Error.Code != "INVALID_INPUT" {
			t.Fatalf("item %d: %+v", index, result)
		}
	}
}

func TestTypedClassificationDefaultBatchBound(t *testing.T) {
	api := newBatchClassificationTestServer(nil)
	body := `{"inputs":[` + strings.TrimSuffix(strings.Repeat(`{"type":"text","text":"x"},`, defaultTypedClassificationBatchLimit+1), ",") + `]}`
	recorder := httptest.NewRecorder()
	api.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
	if recorder.Code != http.StatusBadRequest || !strings.Contains(recorder.Body.String(), "max_batch_size") {
		t.Fatalf("got %d: %s", recorder.Code, recorder.Body.String())
	}
}

// The fake deployment decodes genuine image bytes; the API still uses the
// production prepared binding, capability validation, input conversion and
// model-runtime result validation. No text head accepts an image here.
type typedAPIRuntime struct {
	card       modelservice.ModelCard
	imageCalls atomic.Int32
}

func (r *typedAPIRuntime) Card(context.Context, string) (modelservice.ModelCard, error) {
	return r.card, nil
}
func (r *typedAPIRuntime) CurrentCard(string) (modelservice.ModelCard, bool) { return r.card, true }
func (r *typedAPIRuntime) Classify(context.Context, string, modelservice.ClassifyRequest) (modelservice.ClassifyResponse, error) {
	return modelservice.ClassifyResponse{}, errors.New("unexpected text-head inference")
}

func (r *typedAPIRuntime) Rerank(context.Context, string, modelservice.RerankRequest) (modelservice.RerankResponse, error) {
	return modelservice.RerankResponse{}, errors.New("unexpected reranking")
}

func (r *typedAPIRuntime) Embed(_ context.Context, _ string, request modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	response := modelservice.EmbedResponse{Model: r.card.ID, Embeddings: make([][]float32, len(request.Inputs)), Inputs: make([]*modelservice.InputUsage, len(request.Inputs)), Errors: make([]string, len(request.Inputs))}
	for index, input := range request.Inputs {
		if input.ImageURL != "" {
			r.imageCalls.Add(1)
			if input.Text != "" || input.AudioWAV != "" {
				return modelservice.EmbedResponse{}, errors.New("ambiguous runtime input")
			}
			data, ok := imageurl.DecodeBase64(input.ImageURL)
			if !ok {
				return modelservice.EmbedResponse{}, errors.New("runtime did not receive inline image")
			}
			if _, err := png.Decode(bytes.NewReader(data)); err != nil {
				response.Errors[index] = "invalid image"
				continue
			}
		}
		response.Embeddings[index] = []float32{1, 0}
		response.Inputs[index] = &modelservice.InputUsage{Tokens: 1, ProcessedTokens: 1}
	}
	return response, nil
}

func TestTypedClassificationHTTPUsesExistingImageSignalPath(t *testing.T) {
	deployment := &typedAPIRuntime{card: modelservice.ModelCard{
		ID: "typed-api-images", Repo: "test/typed-api-images", ModelSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(t.TempDir()))), Revision: strings.Repeat("1", 40),
		Surfaces: []string{"embeddings"}, MaxInputTokens: 512, MaxInputs: 8, Device: "cpu", Dtype: "float32", Profile: "exact", Engine: "native",
		Embedding: &modelservice.EmbeddingCard{Dimensions: []int{2}, Modalities: []string{"text", "image"}, Normalized: true, Pooling: "mean"},
	}}
	runtime := serving.New(deployment, nil)
	provider, err := runtime.Embedding(context.Background(), config.ResolvedModelBinding{
		Recipe: config.DefaultRecipeName, Name: "embedding", Binding: config.ModelBinding{Deployment: "vision", Contract: "embedding.v1"},
		Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://runtime.invalid:8100", Device: "cpu", Profile: "exact", Input: config.ModelInputBudget{Overflow: "truncate"}},
	}, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = provider.Close() })
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.EmbeddingConfig.TargetDimension = 2
	cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "image", Candidates: []string{"red square"}, SimilarityThreshold: 0.5, QueryModality: config.QueryModalityImage}, {Name: "text", Candidates: []string{"hello"}, SimilarityThreshold: 0.5}}
	cfg.Decisions = []config.Decision{{Name: "vision", Rules: config.RuleNode{Type: "embedding", Name: "image"}}, {Name: "text", Rules: config.RuleNode{Type: "embedding", Name: "text"}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"vision": {Provider: config.ModelRuntimeProvider, Device: "cpu", Artifact: t.TempDir()}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "vision", Contract: "embedding.v1", Adapter: "vela_omni"}}
	prepared := embedding.NewSet(map[string]embedding.Provider{"mmbert": provider}, "mmbert")
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil, classification.RecipeRuntimeOptions{Embeddings: prepared, Runtime: runtime})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	api := &ClassificationAPIServer{classificationSvc: services.NewRecipeClassificationService(classifiers, cfg), config: cfg}
	var pngBytes bytes.Buffer
	picture := image.NewRGBA(image.Rect(0, 0, 1, 1))
	picture.Set(0, 0, color.RGBA{R: 255, A: 255})
	if err := png.Encode(&pngBytes, picture); err != nil {
		t.Fatal(err)
	}
	uri := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes.Bytes())
	for _, body := range []string{
		fmt.Sprintf(`{"input":{"type":"image","image_url":%q}}`, uri),
		fmt.Sprintf(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q}}]}]}`, uri),
	} {
		recorder := httptest.NewRecorder()
		api.handleIntentClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
		var result services.IntentResponse
		if recorder.Code != http.StatusOK {
			t.Fatalf("got %d: %s", recorder.Code, recorder.Body.String())
		}
		if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Classification.Category != "vision" || result.MatchedSignals == nil || len(result.MatchedSignals.Embeddings) != 1 || result.MatchedSignals.Embeddings[0] != "image" {
			t.Fatalf("did not classify the image: %+v", result)
		}
	}
	body := fmt.Sprintf(`{"inputs":[{"type":"text","text":"hello"},{"type":"image","image_url":%q},{"type":"image","image_url":"https://example.com/not-fetched"}]}`, uri)
	recorder := httptest.NewRecorder()
	api.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("got %d: %s", recorder.Code, recorder.Body.String())
	}
	var result BatchClassificationResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 3 || result.Results[0].Category != "text" || result.Results[1].Category != "vision" || result.Results[2].Error == nil {
		t.Fatalf("unexpected mixed result: %+v", result)
	}
	if deployment.imageCalls.Load() != 1 {
		t.Fatalf("expected one real image dispatch with repeated-image cache reuse: %d", deployment.imageCalls.Load())
	}
	if result.Statistics.CategoryDistribution["vision"] != 1 || result.Statistics.CategoryDistribution["text"] != 1 || len(result.Statistics.CategoryDistribution) != 2 {
		t.Fatalf("failed entries entered statistics: %+v", result.Statistics)
	}
	for _, task := range []string{"pii", "security", "all"} {
		unsupportedBody := fmt.Sprintf(`{"inputs":[{"type":"image","image_url":%q}],"task_type":%q}`, uri, task)
		unsupportedRecorder := httptest.NewRecorder()
		api.handleBatchClassification(unsupportedRecorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(unsupportedBody)))
		var unsupported BatchClassificationResponse
		if err := json.Unmarshal(unsupportedRecorder.Body.Bytes(), &unsupported); err != nil {
			t.Fatal(err)
		}
		if unsupportedRecorder.Code != http.StatusOK || len(unsupported.Results) != 1 || unsupported.Results[0].Error == nil || unsupported.Results[0].Error.Code != "UNSUPPORTED_INPUT" || deployment.imageCalls.Load() != 1 {
			t.Fatalf("%s dispatched an unsupported image: %s", task, unsupportedRecorder.Body.String())
		}
	}
	// Valid base64 with invalid image bytes reaches the genuine image decoder,
	// yields one failure, and still preserves the valid text/image siblings.
	body = fmt.Sprintf(`{"inputs":[{"type":"text","text":"hello"},{"type":"image","image_url":"data:image/png;base64,iVBORw0KGgo="},{"type":"image","image_url":%q}]}`, uri)
	recorder = httptest.NewRecorder()
	api.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
	result = BatchClassificationResponse{}
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusOK || len(result.Results) != 3 || result.Results[0].Error != nil || result.Results[1].Error == nil || result.Results[2].Error != nil || result.Results[2].Category != "vision" || deployment.imageCalls.Load() != 2 {
		t.Fatalf("image decoding failure lost siblings: %s", recorder.Body.String())
	}
}

func TestTypedClassificationOpenAPIVariants(t *testing.T) {
	schema := openAPIRequestSchemaFor[services.ClassificationInput]()
	if len(schema.OneOf) != 2 {
		t.Fatalf("missing variants: %+v", schema)
	}
	for _, variant := range schema.OneOf {
		if len(variant.Required) != 2 || variant.AdditionalProperties != false || len(variant.Properties) != 2 {
			t.Fatalf("ambiguous variant: %+v", variant)
		}
	}
}

func TestTypedSingleTextRetainsLegacyResponse(t *testing.T) {
	api := &ClassificationAPIServer{classificationSvc: services.NewPlaceholderClassificationService()}
	var documents []map[string]interface{}
	for _, body := range []string{`{"text":"hello"}`, `{"input":{"type":"text","text":"hello"}}`} {
		recorder := httptest.NewRecorder()
		api.handleIntentClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
		if recorder.Code != http.StatusOK {
			t.Fatalf("got %d: %s", recorder.Code, recorder.Body.String())
		}
		var document map[string]interface{}
		if err := json.Unmarshal(recorder.Body.Bytes(), &document); err != nil {
			t.Fatal(err)
		}
		document["classification"].(map[string]interface{})["processing_time_ms"] = float64(0)
		documents = append(documents, document)
	}
	left, _ := json.Marshal(documents[0])
	right, _ := json.Marshal(documents[1])
	if !bytes.Equal(left, right) {
		t.Fatalf("typed text changed response: %s != %s", left, right)
	}
	legacy, _ := json.Marshal(BatchClassificationResult{Category: "test", Confidence: 0})
	if string(legacy) != `{"category":"test","confidence":0,"processing_time_ms":0}` {
		t.Fatalf("legacy batch response changed: %s", legacy)
	}
}

func TestTypedClassificationBodyLimit(t *testing.T) {
	api := newBatchClassificationTestServer(nil)
	body := `{"inputs":[{"type":"text","text":"` + strings.Repeat("a", int(defaultJSONRequestBodyLimit)) + `"}]}`
	recorder := httptest.NewRecorder()
	api.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
	if recorder.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected body limit, got %d", recorder.Code)
	}
}

func TestTypedClassificationOpenAPIEnvelopeConstraints(t *testing.T) {
	batch := openAPIRequestSchemaFor[BatchClassificationRequest]()
	if len(batch.OneOf) != 2 || len(batch.OneOf[0].Required) != 1 || batch.OneOf[0].Required[0] != "texts" || len(batch.OneOf[1].Required) != 1 || batch.OneOf[1].Required[0] != "inputs" {
		t.Fatalf("missing exclusive batch inputs: %+v", batch)
	}
	for _, name := range []string{"texts", "inputs"} {
		field := batch.Properties[name]
		if field.Nullable || field.MinItems == nil || *field.MinItems != 1 {
			t.Fatalf("unbounded empty/null %s schema: %+v", name, field)
		}
	}
	intent := openAPIRequestSchemaFor[services.IntentRequest]()
	if len(intent.OneOf) != 2 || intent.OneOf[0].Required[0] != "input" || intent.OneOf[0].Properties["text"].Not == nil || intent.OneOf[0].Properties["messages"].Not == nil || intent.OneOf[1].Not == nil || intent.OneOf[1].Not.Required[0] != "input" {
		t.Fatalf("missing typed/legacy intent exclusion: %+v", intent)
	}
}

type typedBatchResponseFixture struct {
	*services.ClassificationService
	response *services.ClassificationInputsResponse
}

func (s typedBatchResponseFixture) ClassifyInputsContext(context.Context, []services.ClassificationInput, string, *services.IntentOptions) (*services.ClassificationInputsResponse, error) {
	return s.response, nil
}

func TestTypedBatchRetainsUnscoredSuccessAvailability(t *testing.T) {
	available, unavailable := true, false
	service := typedBatchResponseFixture{
		ClassificationService: services.NewPlaceholderClassificationService(),
		response: &services.ClassificationInputsResponse{Results: []services.ClassificationInputResult{
			{Intent: &services.IntentResponse{Classification: services.Classification{Category: "unscored", ConfidenceAvailable: &unavailable}}},
			{Intent: &services.IntentResponse{Classification: services.Classification{Category: "scored", Confidence: .8, ConfidenceAvailable: &available}}},
			{Error: &services.ClassificationItemError{Code: "CLASSIFICATION_ERROR", Message: "classification failed"}},
		}},
	}
	api := &ClassificationAPIServer{}
	recorder := httptest.NewRecorder()
	api.handleTypedBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", nil), service, BatchClassificationRequest{Inputs: make([]services.ClassificationInput, 3), TaskType: "intent"})
	var result BatchClassificationResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusOK || len(result.Results) != 3 {
		t.Fatalf("unexpected response: %s", recorder.Body.String())
	}
	unscored := result.Results[0]
	if unscored.Error != nil || unscored.Category != "unscored" || unscored.Confidence != 0 || unscored.ConfidenceAvailable == nil || *unscored.ConfidenceAvailable {
		t.Fatalf("unavailable score became a measured zero: %+v", unscored)
	}
	if result.Statistics.AvgConfidence != .8 || result.Statistics.LowConfidenceCount != 0 || result.Statistics.CategoryDistribution["unscored"] != 1 || result.Statistics.CategoryDistribution["scored"] != 1 || len(result.Statistics.CategoryDistribution) != 2 {
		t.Fatalf("unscored or failed items distorted statistics: %+v", result.Statistics)
	}
}
