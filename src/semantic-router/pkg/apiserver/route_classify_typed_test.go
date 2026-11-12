//go:build !windows

package apiserver

import (
	"bytes"
	"context"
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

// The fake decision deployment decodes genuine image bytes after the same
// request normalization and capability checks used by the HTTP handlers.
type typedAPIRuntime struct{ imageCalls atomic.Int32 }

func (*typedAPIRuntime) Card(context.Context, string) (modelservice.ModelCard, error) {
	return modelservice.ModelCard{ID: "fixture", Surfaces: []string{"decisions"}, QuestionTypes: []string{"noul"}, Modalities: []string{"text", "image"}}, nil
}

func (r *typedAPIRuntime) CurrentCard(name string) (modelservice.ModelCard, bool) {
	card, _ := r.Card(context.Background(), name)
	return card, true
}

func (*typedAPIRuntime) Embed(context.Context, string, modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	return modelservice.EmbedResponse{}, errors.New("unexpected embedding call")
}

func (r *typedAPIRuntime) Decide(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	for _, uri := range request.Images {
		r.imageCalls.Add(1)
		data, ok := imageurl.DecodeBase64(uri)
		if !ok {
			return modelservice.Response{}, errors.New("runtime did not receive inline image")
		}
		if _, err := png.Decode(bytes.NewReader(data)); err != nil {
			return modelservice.Response{}, errors.New("invalid image")
		}
	}
	answers := make(map[string]modelservice.Answer)
	for _, question := range request.Questions {
		answers[question.ID] = modelservice.Answer{Type: "noul", Noul: 0.9, InputCoverage: "complete"}
	}
	return modelservice.Response{Answers: answers}, nil
}

const typedAPIImageConfig = `version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
recipes:
  - name: default
    signals:
      systemone:
        - name: image
          input: {image: true}
          question: {type: noul, instructions: "Is this an image?"}
    decisions:
      - name: vision
        priority: 100
        rules:
          operator: AND
          conditions: [{type: systemone, name: image}]
        modelRefs: [{model: general}]
global:
  model_catalog:
    system: {decision_model: {deployment: primary}}
    deployments:
      primary: {provider: model_runtime, artifact: /models/custom-image, device: cpu}
`

func TestTypedClassificationHTTPUsesExistingImageSignalPath(t *testing.T) {
	deployment := &typedAPIRuntime{}
	cfg, err := config.ParseYAMLBytes([]byte(typedAPIImageConfig))
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, classification.RecipeRuntimeOptions{Runtime: serving.New(deployment, nil)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	if err := classifiers.InitializeRuntime(); err != nil {
		t.Fatal(err)
	}
	api := &ClassificationAPIServer{classificationSvc: services.NewRecipeClassificationService(classifiers, cfg), config: cfg}
	var pngBytes bytes.Buffer
	picture := image.NewRGBA(image.Rect(0, 0, 1, 1))
	picture.Set(0, 0, color.RGBA{R: 255, A: 255})
	if err := png.Encode(&pngBytes, picture); err != nil {
		t.Fatal(err)
	}
	uri := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes.Bytes())
	for _, body := range []string{fmt.Sprintf(`{"input":{"type":"image","image_url":%q}}`, uri), fmt.Sprintf(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":%q}}]}]}`, uri)} {
		recorder := httptest.NewRecorder()
		api.handleIntentClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
		var result services.IntentResponse
		if recorder.Code != http.StatusOK {
			t.Fatalf("status %d: %s", recorder.Code, recorder.Body.String())
		}
		if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Classification.Category != "vision" {
			t.Fatalf("did not classify image: %+v", result)
		}
	}
	body := fmt.Sprintf(`{"inputs":[{"type":"text","text":"hello"},{"type":"image","image_url":%q},{"type":"image","image_url":"https://example.com/not-fetched"}]}`, uri)
	recorder := httptest.NewRecorder()
	api.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
	var result BatchClassificationResponse
	if recorder.Code != http.StatusOK {
		t.Fatalf("status %d: %s", recorder.Code, recorder.Body.String())
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Results) != 3 || result.Results[0].Error != nil || result.Results[1].Category != "vision" || result.Results[2].Error == nil {
		t.Fatalf("mixed result: %+v", result)
	}
	if deployment.imageCalls.Load() != 3 {
		t.Fatalf("expected three image dispatches: %d", deployment.imageCalls.Load())
	}
	for _, task := range []string{"pii", "security", "all"} {
		body = fmt.Sprintf(`{"inputs":[{"type":"image","image_url":%q}],"task_type":%q}`, uri, task)
		recorder = httptest.NewRecorder()
		api.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
		var unsupported BatchClassificationResponse
		if err := json.Unmarshal(recorder.Body.Bytes(), &unsupported); err != nil {
			t.Fatal(err)
		}
		if recorder.Code != http.StatusOK || len(unsupported.Results) != 1 || unsupported.Results[0].Error == nil || unsupported.Results[0].Error.Code != "UNSUPPORTED_INPUT" || deployment.imageCalls.Load() != 3 {
			t.Fatalf("unsupported %s: %s", task, recorder.Body.String())
		}
	}
	body = fmt.Sprintf(`{"inputs":[{"type":"text","text":"hello"},{"type":"image","image_url":"data:image/png;base64,iVBORw0KGgo="},{"type":"image","image_url":%q}]}`, uri)
	recorder = httptest.NewRecorder()
	api.handleBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
	result = BatchClassificationResponse{}
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusOK || len(result.Results) != 3 || result.Results[0].Error != nil || result.Results[1].Error == nil || result.Results[2].Error != nil || result.Results[2].Category != "vision" || deployment.imageCalls.Load() != 5 {
		t.Fatalf("decoder failure lost siblings: %s", recorder.Body.String())
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
