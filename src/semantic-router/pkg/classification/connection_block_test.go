package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector/connectortest"
)

func TestRemoteClassifierDeploymentsConnectThroughTheirConnectionBlock(t *testing.T) {
	gateway := connectortest.New(t, map[string]http.HandlerFunc{
		"/classify": func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`[{"label":"other","score":0.25},{"label":"math","score":0.75}]`))
		},
		"/chat/completions": func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"c1","object":"chat.completion","created":1,"model":"judge-model","choices":[{"index":0,"message":{"role":"assistant","content":"SAFE"},"finish_reason":"stop"}]}`))
		},
	})
	ctx := context.Background()

	classify, err := newHTTPClassifierInference(gateway.Deployment(config.ClassifyProvider, ""), newDeclaredLabelMapping([]string{"math", "other"}))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classify.Close() })
	result, err := classify.Classify(ctx, "What is 2 + 2?")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Probabilities) != 2 || result.Probabilities[0] != 0.75 {
		t.Fatalf("labels read by position, not by name: %+v", result)
	}

	chat := newVLLMClientForDeployment(gateway.Deployment(config.OpenAIChatProvider, "judge-model"))
	t.Cleanup(func() { _ = chat.Close() })
	completion, err := chat.Generate(ctx, "judge-model", "Is this request safe?", &GenerationOptions{MaxTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	if len(completion.Choices) != 1 || completion.Choices[0].Message.Content != "SAFE" {
		t.Fatalf("completion: %+v", completion)
	}

	paths := map[string]map[string]json.RawMessage{}
	for _, call := range gateway.Calls() {
		var body map[string]json.RawMessage
		_ = json.Unmarshal(call.Body, &body)
		paths[call.Path] = body
	}
	if _, ok := paths[connectortest.Prefix+"/classify"]; !ok {
		t.Fatalf("classify did not reach %s/classify: %v", connectortest.Prefix, paths)
	}
	if body, ok := paths[connectortest.Prefix+"/chat/completions"]; !ok || string(body["model"]) != `"judge-model"` {
		t.Fatalf("chat did not ask judge-model under %s: %v", connectortest.Prefix, paths)
	}
}
