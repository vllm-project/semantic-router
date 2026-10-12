package classification

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The LLM label classifier must pin its reply to the parsed contract with a
// strict json_schema response_format, not just prompt instructions: a small
// model that emits malformed JSON otherwise silently voids the signal.
func TestLLMLabelClassifierRequestsStrictJSONSchema(t *testing.T) {
	var requestFormat map[string]interface{}
	requestReady := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		var payload map[string]interface{}
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode request payload: %v", err)
		}
		requestFormat, _ = payload["response_format"].(map[string]interface{})
		close(requestReady)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"scores\":{\"math\":0.9,\"other\":0.1},\"rationale\":\"clear\"}"}}]}`))
	}))
	defer server.Close()

	client := newVLLMClientFromConfig(&config.ExternalModelConfig{
		ModelName:     "label-model",
		ModelEndpoint: endpointForTestServer(t, server),
	})
	if client.initErr != nil {
		t.Fatalf("newVLLMClientFromConfig: %v", client.initErr)
	}
	classifier := &llmLabelClassifier{
		client:       client,
		model:        "label-model",
		labels:       []string{"math", "other"},
		instructions: "Score the prompt.",
		timeout:      5 * time.Second,
		maxTokens:    256,
	}
	classification, err := classifier.classify(context.Background(), "solve this")
	if err != nil {
		t.Fatalf("classify: %v", err)
	}
	if classification.Scores["math"] != 0.9 || classification.Scores["other"] != 0.1 {
		t.Fatalf("classification = %#v", classification)
	}

	select {
	case <-requestReady:
	default:
		t.Fatal("classifier request was not captured")
	}
	if requestFormat == nil {
		t.Fatal("request must carry a response_format")
	}
	if got, _ := requestFormat["type"].(string); got != "json_schema" {
		t.Fatalf("response_format type = %q, want json_schema", got)
	}
	schemaSpec, _ := requestFormat["json_schema"].(map[string]interface{})
	if schemaSpec == nil {
		t.Fatal("response_format must carry a json_schema object")
	}
	if strict, ok := schemaSpec["strict"].(bool); !ok || !strict {
		t.Fatalf("json_schema strict = %#v, want true", schemaSpec["strict"])
	}
	schema, _ := schemaSpec["schema"].(map[string]interface{})
	if schema == nil {
		t.Fatal("json_schema must carry a schema")
	}
	scores, _ := schema["properties"].(map[string]interface{})["scores"].(map[string]interface{})
	if scores == nil {
		t.Fatal("schema must constrain the scores object")
	}
	scoreProperties, _ := scores["properties"].(map[string]interface{})
	for _, label := range []string{"math", "other"} {
		if _, ok := scoreProperties[label]; !ok {
			t.Fatalf("scores schema must constrain label %q: %#v", label, scoreProperties)
		}
	}
	requiredLabels, _ := scores["required"].([]interface{})
	if len(requiredLabels) != 2 {
		t.Fatalf("scores schema must require every label, got %#v", requiredLabels)
	}
}

func TestLabelScoresJSONSchemaOmitsRationaleWhenDisabled(t *testing.T) {
	withRationale := labelScoresJSONSchema([]string{"a", "b"}, false)
	if _, ok := withRationale.Schema["properties"].(map[string]interface{})["rationale"]; !ok {
		t.Fatal("rationale must be present when not disabled")
	}
	withoutRationale := labelScoresJSONSchema([]string{"a", "b"}, true)
	if _, ok := withoutRationale.Schema["properties"].(map[string]interface{})["rationale"]; ok {
		t.Fatal("rationale must be omitted when disabled")
	}
}
