package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const looperRatingsProbeKeyword = "__LOOPER_RATINGS_PROBE__"

type ratingsMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ratingsChoice struct {
	Index        int            `json:"index"`
	Model        string         `json:"model"`
	FinishReason string         `json:"finish_reason"`
	Message      ratingsMessage `json:"message"`
}

type ratingsCompletion struct {
	Object  string          `json:"object"`
	Model   string          `json:"model"`
	Choices []ratingsChoice `json:"choices"`
	Usage   struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

func init() {
	pkgtestcases.Register("looper-ratings-happy-path", pkgtestcases.TestCase{
		Description: "Verify deterministic Ratings fan-out, ordered choices, and aggregate usage",
		Tags:        []string{"kubernetes", "routing", "looper", "ratings"},
		Fn:          testLooperRatingsHappyPath,
	})
}

func testLooperRatingsHappyPath(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", looperRatingsProbeKeyword, 30*time.Second)
	if err != nil {
		return fmt.Errorf("ratings request failed: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-ratings-happy-path")
		return fmt.Errorf("ratings request: %s", formatUnexpectedChatCompletionStatus(response))
	}

	expectedHeaders := map[string]string{
		"x-vsr-selected-decision":        "looper_ratings_decision",
		"x-vsr-response-path":            "looper",
		"x-vsr-looper-algorithm":         "ratings",
		"x-vsr-looper-models-used":       "ratings-model-a,ratings-model-b",
		"x-vsr-looper-iterations":        "2",
		"x-vsr-looper-prompt-tokens":     "10",
		"x-vsr-looper-completion-tokens": "6",
		"x-vsr-looper-total-tokens":      "16",
	}
	for name, want := range expectedHeaders {
		if got := response.Headers.Get(name); got != want {
			return fmt.Errorf("header %s = %q, want %q", name, got, want)
		}
	}

	var completion ratingsCompletion
	if err := json.Unmarshal(response.Body, &completion); err != nil {
		return fmt.Errorf("decode ratings response: %w", err)
	}
	if completion.Object != "chat.completion" {
		return fmt.Errorf("response object = %q, want chat.completion", completion.Object)
	}
	if completion.Model != "ratings-model-a,ratings-model-b" {
		return fmt.Errorf("response model = %q, want ratings-model-a,ratings-model-b", completion.Model)
	}

	wantChoices := []ratingsChoice{
		{Index: 0, Model: "ratings-model-a", FinishReason: "stop", Message: ratingsMessage{Role: "assistant", Content: "answer-from-ratings-model-a"}},
		{Index: 1, Model: "ratings-model-b", FinishReason: "stop", Message: ratingsMessage{Role: "assistant", Content: "answer-from-ratings-model-b"}},
	}
	if !reflect.DeepEqual(completion.Choices, wantChoices) {
		return fmt.Errorf("ratings choices = %+v, want %+v", completion.Choices, wantChoices)
	}
	if got := []int64{completion.Usage.PromptTokens, completion.Usage.CompletionTokens, completion.Usage.TotalTokens}; !reflect.DeepEqual(got, []int64{10, 6, 16}) {
		return fmt.Errorf("ratings usage = %v, want [10 6 16]", got)
	}

	return nil
}
