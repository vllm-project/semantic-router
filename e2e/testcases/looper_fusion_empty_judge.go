package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const looperFusionEmptyJudgeProbeKeyword = "__LOOPER_FUSION_EMPTY_JUDGE_PROBE__"

func init() {
	pkgtestcases.Register("looper-fusion-empty-final-judge", pkgtestcases.TestCase{
		Description: "Reject an empty final Fusion judge answer through Envoy",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionEmptyFinalJudge,
	})
}

func testLooperFusionEmptyFinalJudge(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	counters, err := openFusionCounters(ctx, client, opts)
	if err != nil {
		return err
	}
	defer counters.close()

	response, err := sendLocalChatCompletion(ctx, localPort, "vllm-sr/auto", looperFusionEmptyJudgeProbeKeyword, 30*time.Second)
	if err != nil {
		return fmt.Errorf("empty final Fusion judge request failed: %w", err)
	}
	if response.StatusCode != http.StatusInternalServerError {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-fusion-empty-final-judge")
		return fmt.Errorf("empty final Fusion judge status = %d, want %d", response.StatusCode, http.StatusInternalServerError)
	}

	var payload looperFusionErrorResponse
	if err := json.Unmarshal(response.Body, &payload); err != nil {
		return fmt.Errorf("decode empty final Fusion judge error: %w", err)
	}
	if !strings.Contains(payload.Error.Message, "empty final answer") {
		return fmt.Errorf("empty final Fusion judge error = %q, want an empty-answer error", payload.Error.Message)
	}

	return counters.requireCounts(ctx, "looper-fusion-empty-final-judge", map[string]int{
		"fusion-panel-valid":   1,
		looperFusionJudgeModel: 1,
	})
}
