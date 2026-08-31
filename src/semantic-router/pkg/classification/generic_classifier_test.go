package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLLMLabelClassifierMaxTokens(t *testing.T) {
	tests := []struct {
		name       string
		configured int
		want       int
	}{
		{name: "configured", configured: 384, want: 384},
		{name: "default", want: defaultLLMLabelClassifierMaxTokens},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requests := make(chan int, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					MaxTokens int `json:"max_tokens"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				requests <- request.MaxTokens
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"choices": []map[string]interface{}{{
						"message": map[string]interface{}{
							"content": `{"label":"safe","rationale":"allowed"}`,
						},
					}},
				})
			}))
			defer server.Close()

			classifier, err := newLLMLabelClassifier(
				config.ClassifierSignalRule{
					Model:        "test-model",
					Labels:       []string{"safe", "unsafe"},
					Instructions: "Classify the input.",
				},
				&config.ExternalModelConfig{
					ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "placeholder", Port: 1},
					ModelName:     "test-model",
					MaxTokens:     tt.configured,
				},
			)
			if err != nil {
				t.Fatalf("newLLMLabelClassifier() error = %v", err)
			}
			classifier.(*llmLabelClassifier).client.baseURL = server.URL

			if _, err := classifier.Classify(context.Background(), "hello"); err != nil {
				t.Fatalf("Classify() error = %v", err)
			}
			if got := <-requests; got != tt.want {
				t.Errorf("max_tokens = %d, want %d", got, tt.want)
			}
		})
	}
}
