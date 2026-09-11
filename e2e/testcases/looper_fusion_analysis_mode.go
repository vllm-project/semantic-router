package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

type looperFusionModeResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
	Fusion json.RawMessage `json:"fusion"`
}

func init() {
	pkgtestcases.Register("looper-fusion-analysis-modes", pkgtestcases.TestCase{
		Description: "Validate Fusion mode answers, accounting, iterations, and public trace suppression",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"},
		Fn:          testLooperFusionAnalysisModes,
	})
}

func testLooperFusionAnalysisModes(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	tests := []struct {
		name            string
		probe           string
		wantAnswer      string
		wantTotalTokens int
		wantIterations  string
	}{
		{
			name:            "separate",
			probe:           "__LOOPER_FUSION_SEPARATE_PROBE__",
			wantAnswer:      "fusion-separate-answer",
			wantTotalTokens: 58,
			wantIterations:  "3",
		},
		{
			name:            "one_call",
			probe:           "__LOOPER_FUSION_ONE_CALL_PROBE__",
			wantAnswer:      "fusion-one-call-answer",
			wantTotalTokens: 36,
			wantIterations:  "2",
		},
		{
			name:            "none",
			probe:           "__LOOPER_FUSION_NONE_PROBE__",
			wantAnswer:      "fusion-none-answer",
			wantTotalTokens: 36,
			wantIterations:  "2",
		},
	}

	for _, tc := range tests {
		response, err := sendLocalChatCompletion(ctx, localPort, "MoM", tc.probe, 30*time.Second)
		if err != nil {
			return fmt.Errorf("fusion %s request failed: %w", tc.name, err)
		}
		if response.StatusCode != http.StatusOK {
			logUnexpectedChatCompletionStatus(opts.Verbose, response, "looper-fusion-analysis-modes")
			return fmt.Errorf("fusion %s status = %d, want %d", tc.name, response.StatusCode, http.StatusOK)
		}

		var payload looperFusionModeResponse
		if err := json.Unmarshal(response.Body, &payload); err != nil {
			return fmt.Errorf("decode fusion %s response: %w", tc.name, err)
		}
		if len(payload.Choices) != 1 || payload.Choices[0].Message.Content != tc.wantAnswer {
			return fmt.Errorf("fusion %s answer = %#v, want %q", tc.name, payload.Choices, tc.wantAnswer)
		}
		if got := response.Headers.Get("x-vsr-looper-iterations"); got != tc.wantIterations {
			return fmt.Errorf("fusion %s iterations header = %q, want %q", tc.name, got, tc.wantIterations)
		}
		if payload.Usage.TotalTokens != tc.wantTotalTokens {
			return fmt.Errorf("fusion %s total_tokens = %d, want %d", tc.name, payload.Usage.TotalTokens, tc.wantTotalTokens)
		}
		if len(payload.Fusion) != 0 {
			return fmt.Errorf("fusion %s response exposed a mode-only fusion member: %s", tc.name, payload.Fusion)
		}
	}

	return nil
}
