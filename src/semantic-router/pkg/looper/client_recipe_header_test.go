package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/internalauth"
)

func TestExecuteWithLatencyPropagatesParentRecipeHeader(t *testing.T) {
	receivedRecipe := make(chan string, 1)
	receivedAuthenticated := make(chan bool, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		receivedRecipe <- request.Header.Get(headers.VSRSelectedRecipe)
		receivedAuthenticated <- internalauth.Authenticate(request.Header.Get(headers.VSRInternalAuth))
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-recipe",
			"object":  "chat.completion",
			"created": 1,
			"model":   "model-a",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "ok",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	request := &Request{
		OriginalRequest: &openai.ChatCompletionNewParams{
			Model:    "entrypoint",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
		},
		ModelRefs:    []config.ModelRef{{Model: "model-a"}},
		DecisionName: "shared-route",
		RecipeName:   "named-recipe",
	}
	response, err := ExecuteWithLatency(
		context.Background(),
		NewBaseLooper(&config.LooperConfig{
			Endpoint: server.URL,
			Headers: map[string]string{
				headers.VSRInternalAuth: "configured-value-must-not-override",
			},
		}),
		request,
	)
	if err != nil {
		t.Fatalf("ExecuteWithLatency: %v", err)
	}
	if response == nil {
		t.Fatal("expected looper response")
	}
	if got := <-receivedRecipe; got != "named-recipe" {
		t.Fatalf("%s = %q, want %q", headers.VSRSelectedRecipe, got, "named-recipe")
	}
	if !<-receivedAuthenticated {
		t.Fatal("looper client did not authenticate its internal request context")
	}
}
