package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLLMClassifierReasoningResponseThroughClient(t *testing.T) {
	for _, tc := range []struct{ name, content string }{
		{"plain_json_control", `{"scores":{"A":0.2,"B":0.8},"rationale":"label B"}`},
		{"reasoning_in_content", `<think>Compare both labels.</think>
{"scores":{"A":0.2,"B":0.8},"rationale":"label B"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var sent map[string]json.RawMessage
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&sent); err != nil {
					t.Error(err)
				}
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": tc.content}}}})
			}))
			defer server.Close()
			c, err := newLLMLabelClassifier(config.ClassifierSignalRule{Name: "labels", Model: "test", Labels: []string{"A", "B"}, Instructions: "Classify this benign text."}, &config.ExternalModelConfig{ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "placeholder", Port: 1}, ModelName: "test"}, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer c.(*llmLabelClassifier).Close()
			setTestVLLMClientURL(c.(*llmLabelClassifier).client, server.URL)
			result, err := c.Classify(context.Background(), "compare labels")
			t.Logf("response_format=%s actual_scores=%v error=%v", sent["response_format"], result.Scores, err)
			if tc.name == "plain_json_control" {
				if err != nil || result.Scores["B"] != 0.8 {
					t.Fatalf("plain JSON control failed: %v %v", result, err)
				}
			} else if err != nil {
				t.Errorf("reasoning followed by valid final JSON should classify, got %v", err)
			}
		})
	}
}

func TestParseLLMLabelClassificationReasoningBoundaries(t *testing.T) {
	const answer = `{"scores":{"A":0.2,"B":0.8},"rationale":"label B"}`
	for _, tc := range []struct {
		name, content string
		wantError     bool
	}{
		{"multiple complete blocks", " <think>First.</think>\n<think>Second.</think>\n" + answer, false},
		{"JSON inside reasoning", `<think>{"scores":{"A":1,"B":0},"rationale":"draft"}</think>` + answer, false},
		{"literal tags in rationale", `{"scores":{"A":0.2,"B":0.8},"rationale":"literal <think>text</think>"}`, false},
		{"unterminated block", "<think>Not finished " + answer, true},
		{"missing final answer", "<think>Only reasoning.</think>", true},
		{"invalid scores", `<think>Complete.</think>{"scores":{"A":2,"B":-1},"rationale":"invalid"}`, true},
		{"extra trailing text", answer + " trailing", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseLLMLabelClassification(tc.content, []string{"A", "B"}, false)
			if (err != nil) != tc.wantError {
				t.Fatalf("result=%+v error=%v", got, err)
			}
			if err == nil && got.Scores["B"] != 0.8 {
				t.Fatalf("wrong final scores: %+v", got)
			}
		})
	}
}
