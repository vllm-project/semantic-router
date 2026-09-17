package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLLMLabelClassifierReturnsReportedDistribution(t *testing.T) {
	const answer = `{"scores":{"SAFE":0.23,"RISKY":0.77},"rationale":"destructive operation"}`
	const literalRationale = "literal <think>example</think> and </think> text"
	tests := []struct {
		name      string
		content   string
		rationale string
	}{
		{
			name:      "plain JSON",
			content:   answer,
			rationale: "destructive operation",
		},
		{
			name:      "reasoning prefix",
			content:   "<think>Evaluate the operation.</think>\n" + answer,
			rationale: "destructive operation",
		},
		{
			name:      "misleading JSON in reasoning",
			content:   `<think>{"scores":{"SAFE":1,"RISKY":0},"rationale":"draft"}</think>` + answer,
			rationale: "destructive operation",
		},
		{
			name:      "multiple blocks and whitespace",
			content:   " \n<think>First.\nSecond.</think>\n<think>Third.</think>\n" + answer + "\n",
			rationale: "destructive operation",
		},
		{
			name:      "preserves tags in JSON",
			content:   `{"scores":{"SAFE":0.23,"RISKY":0.77},"rationale":"` + literalRationale + `"}`,
			rationale: literalRationale,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Messages []struct {
						Content string `json:"content"`
					} `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				if len(request.Messages) != 2 || !strings.Contains(request.Messages[0].Content, `"scores"`) {
					http.Error(w, "missing score contract", http.StatusBadRequest)
					return
				}
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"choices": []map[string]interface{}{{
						"message": map[string]interface{}{
							"content": tt.content,
						},
					}},
				})
			}))
			defer server.Close()

			classifier, err := newLLMLabelClassifier(
				config.ClassifierSignalRule{
					Model:        "test-model",
					Labels:       []string{"SAFE", "RISKY"},
					Instructions: "Classify the input.",
				},
				&config.ExternalModelConfig{
					ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "placeholder", Port: 1},
					ModelName:     "test-model",
				},
			)
			if err != nil {
				t.Fatalf("newLLMLabelClassifier() error = %v", err)
			}
			defer classifier.(*llmLabelClassifier).Close()
			setTestVLLMClientURL(classifier.(*llmLabelClassifier).client, server.URL)

			result, err := classifier.Classify(context.Background(), "delete production")
			if err != nil {
				t.Fatalf("Classify() error = %v", err)
			}
			if result.Scores["SAFE"] != 0.23 || result.Scores["RISKY"] != 0.77 {
				t.Errorf("scores = %v, want SAFE=0.23 and RISKY=0.77", result.Scores)
			}
			if result.Rationale != tt.rationale {
				t.Errorf("rationale = %q, want %q", result.Rationale, tt.rationale)
			}
		})
	}
}

func TestParseLLMLabelClassificationRejectsInvalidDistribution(t *testing.T) {
	const answer = `{"scores":{"SAFE":0.23,"RISKY":0.77},"rationale":"destructive operation"}`
	tests := []struct {
		name    string
		content string
		wantErr string
	}{
		{
			name:    "legacy one-hot response",
			content: `{"label":"RISKY","rationale":"test"}`,
			wantErr: "exactly scores and rationale",
		},
		{
			name:    "missing label",
			content: `{"scores":{"RISKY":1},"rationale":"test"}`,
			wantErr: "exactly the declared labels",
		},
		{
			name:    "undeclared label",
			content: `{"scores":{"SAFE":0.2,"OTHER":0.8},"rationale":"test"}`,
			wantErr: `missing label "RISKY"`,
		},
		{
			name:    "score out of range",
			content: `{"scores":{"SAFE":-0.1,"RISKY":1.1},"rationale":"test"}`,
			wantErr: "within [0, 1]",
		},
		{
			name:    "scores do not sum to one",
			content: `{"scores":{"SAFE":0.2,"RISKY":0.3},"rationale":"test"}`,
			wantErr: "want approximately 1",
		},
		{
			name:    "empty rationale",
			content: `{"scores":{"SAFE":0.2,"RISKY":0.8},"rationale":" "}`,
			wantErr: "empty rationale",
		},
		{
			name:    "unclosed reasoning",
			content: "<think>" + answer,
			wantErr: "unterminated reasoning block",
		},
		{
			name:    "no final answer",
			content: "<think>Done.</think>",
			wantErr: "invalid JSON",
		},
		{
			name:    "unmarked prose",
			content: "Let me think. " + answer,
			wantErr: "invalid JSON",
		},
		{
			name:    "trailing prose",
			content: "<think>Done.</think>" + answer + " explanation",
			wantErr: "invalid JSON",
		},
		{
			name:    "multiple answers",
			content: "<think>Done.</think>" + answer + answer,
			wantErr: "invalid JSON",
		},
		{
			name:    "invalid final scores",
			content: `<think>Done.</think>{"scores":{"SAFE":0.1,"RISKY":0.2},"rationale":"invalid sum"}`,
			wantErr: "want approximately 1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseLLMLabelClassification(tt.content, []string{"SAFE", "RISKY"}, false)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("parseLLMLabelClassification() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}
