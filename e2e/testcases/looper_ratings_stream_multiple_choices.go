package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const looperRatingsDecision = "looper_ratings_decision"

func init() {
	pkgtestcases.Register("looper-ratings-stream-multiple-choices", pkgtestcases.TestCase{
		Description: "A streamed request for several choices under Ratings gets the typed 400 a single-model route gets, with no backend call, while several buffered choices still fan out (issue #5574)",
		Tags:        []string{"kubernetes", "routing", "looper", "ratings", "streaming", "failure"},
		Fn:          testLooperRatingsStreamMultipleChoices,
	})
}

// Ratings sends one hop per model, and dispatch refuses each hop of a streamed
// n>1 request before any backend call. The run used to answer 500
// request_failed, which a client retries although no retry can succeed.
func testLooperRatingsStreamMultipleChoices(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	counters, err := openFusionCounters(ctx, client, opts)
	if err != nil {
		return err
	}
	defer counters.close()

	// The control runs first: it proves the request reaches Ratings and the
	// backend counters are live, so the refused request's zero counts mean no
	// dispatch.
	control, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", looperRatingsCandidatesRequest(2, false), false, nil)
	if err != nil {
		return fmt.Errorf("buffered n=2 request: %w", err)
	}
	if control.StatusCode != http.StatusOK {
		return fmt.Errorf("buffered n=2 returned HTTP %d, want 200: %s", control.StatusCode, truncateString(string(control.Body), 500))
	}
	if got := control.Headers.Get("x-vsr-selected-decision"); got != looperRatingsDecision {
		return fmt.Errorf("buffered n=2 selected decision %q, want %s", got, looperRatingsDecision)
	}
	if countErr := counters.requireCounts(ctx, "buffered n=2", map[string]int{"ratings-model-a": 1, "ratings-model-b": 1}); countErr != nil {
		return countErr
	}
	if resetErr := counters.reset(ctx); resetErr != nil {
		return resetErr
	}

	result, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", looperRatingsCandidatesRequest(2, true), true, nil)
	if err != nil {
		return fmt.Errorf("streamed n=2 request: %w", err)
	}
	if result.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("streamed n=2 returned HTTP %d, want 400: %s", result.StatusCode, truncateString(string(result.Body), 500))
	}
	if got := result.Headers.Get("x-vsr-selected-decision"); got != looperRatingsDecision {
		return fmt.Errorf("streamed n=2 selected decision %q, want %s", got, looperRatingsDecision)
	}
	var response struct {
		Error struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
	}
	if decodeErr := json.Unmarshal(result.Body, &response); decodeErr != nil {
		return fmt.Errorf("decode streamed n=2 error: %w: %s", decodeErr, truncateString(string(result.Body), 500))
	}
	if response.Error.Type != "invalid_request_error" || response.Error.Code != "stream_multiple_choices" {
		return fmt.Errorf("streamed n=2 returned the wrong error: %s", truncateString(string(result.Body), 500))
	}
	return counters.requireCounts(ctx, "streamed n=2", map[string]int{"ratings-model-a": 0, "ratings-model-b": 0})
}

func looperRatingsCandidatesRequest(n int, stream bool) map[string]any {
	return map[string]any{
		"model": "vllm-sr/auto", "n": n, "stream": stream,
		"messages": []map[string]string{{"role": "user", "content": looperRatingsProbeKeyword}},
	}
}
