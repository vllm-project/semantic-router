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

const looperFusionSynthesisProbeKeyword = "__LOOPER_FUSION_SYNTHESIS_TRACE_PROBE__"

func init() {
	pkgtestcases.Register("looper-fusion-synthesis-trace", pkgtestcases.TestCase{
		Description: "Serve a Fusion synthesis with default trace visibility and partial panel failure",
		Tags:        []string{"kubernetes", "routing", "looper", "fusion"}, Fn: testLooperFusionSynthesisTrace,
	})
}

func testLooperFusionSynthesisTrace(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()
	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", looperFusionSynthesisProbeKeyword, 60*time.Second)
	if err != nil {
		return fmt.Errorf("request traced synthesis: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("traced synthesis status=%d, want200: %s", response.StatusCode, response.Body)
	}
	return validateLooperSynthesisTrace(response.Body)
}

func validateLooperSynthesisTrace(body []byte) error {
	var response struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Fusion struct {
			Responses []struct {
				Model string `json:"model"`
			} `json:"responses"`
			FailedModels []struct {
				Model string `json:"model"`
			} `json:"failed_models"`
		} `json:"fusion"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return fmt.Errorf("decode traced synthesis: %w", err)
	}
	if response.Object != "chat.completion" || len(response.Choices) != 1 || response.Choices[0].Message.Content != "fusion-none-answer" {
		return fmt.Errorf("traced synthesis did not return the expected final answer")
	}
	if len(response.Fusion.Responses) != 1 || response.Fusion.Responses[0].Model != "fusion-panel-valid" {
		return fmt.Errorf("traced synthesis lost the usable panel evidence")
	}
	if len(response.Fusion.FailedModels) != 1 || response.Fusion.FailedModels[0].Model != "fusion-panel-fail" {
		return fmt.Errorf("traced synthesis lost the failed panel evidence")
	}
	return nil
}
