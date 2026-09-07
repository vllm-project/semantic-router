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

const (
	looperFusionQuorumProbeKeyword = "__LOOPER_FUSION_QUORUM_PROBE__"
	looperFusionSynthesizedAnswer  = "unexpected-fusion-synthesized-answer"
)

type looperFusionErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func init() {
	pkgtestcases.Register("looper-fusion-usable-quorum", pkgtestcases.TestCase{
		Description: "Reject Fusion synthesis when usable panel responses do not meet quorum",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionUsableQuorum,
	})
}

func testLooperFusionUsableQuorum(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", looperFusionQuorumProbeKeyword, 30*time.Second)
	if err != nil {
		return fmt.Errorf("fusion quorum request failed: %w", err)
	}
	if response.StatusCode != http.StatusInternalServerError {
		logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-fusion-usable-quorum")
		return fmt.Errorf("fusion quorum status = %d, want %d", response.StatusCode, http.StatusInternalServerError)
	}

	var errorResponse looperFusionErrorResponse
	if err := json.Unmarshal(response.Body, &errorResponse); err != nil {
		return fmt.Errorf("decode fusion quorum error: %w", err)
	}
	if errorResponse.Error.Type != "server_error" {
		return fmt.Errorf("error.type = %q, want server_error: %s", errorResponse.Error.Type, string(response.Body))
	}
	if strings.TrimSpace(errorResponse.Error.Message) == "" {
		return fmt.Errorf("error.message is empty: %s", string(response.Body))
	}
	if strings.Contains(string(response.Body), looperFusionSynthesizedAnswer) {
		return fmt.Errorf("fusion quorum error unexpectedly returned synthesized answer %q", looperFusionSynthesizedAnswer)
	}

	return nil
}
