package testcases

import (
	"context"
	"fmt"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("jailbreak-onerror-block", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P0,
		Description: "Test explicit unknown no-match and the fail-closed jailbreak signal against an unreachable deployment",
		Tags:        []string{"kubernetes", "security", "jailbreak", "prompt-guard"},
		Fn:          testJailbreakOnErrorBlock,
	})
}

// testJailbreakOnErrorBlock sends an ordinary, non-adversarial prompt against
// the jailbreak-onerror profile (its jailbreak rule's systemone deployment
// points at an endpoint nothing listens on). It must be blocked: with the
// deployment unreachable, the rule's on_unknown: match counts the unknown
// jailbreak signal as a detection rather than letting a request through
// unchecked.
func testJailbreakOnErrorBlock(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing the fail-closed jailbreak signal against an unreachable deployment")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	const benignPrompt = "What is the capital of France?"

	// Must use the auto-routing model name, not a concrete backend model
	// like "openai/mock-model" directly: concrete model IDs are passthrough
	// requests that bypass every recipe-local signal, decision, and plugin
	// (see performDecisionEvaluation), so they'd never reach the jailbreak
	// signal being tested here.
	response, err := sendLocalChatCompletion(ctx, localPort, "vllm-sr/auto", benignPrompt, 30*time.Second)
	if err != nil {
		return fmt.Errorf("chat completion request failed: %w", err)
	}

	responsePath := response.Headers.Get(responsePathHeader)
	selectedDecision := response.Headers.Get("x-vsr-selected-decision")
	blocked := responsePath == "fast_response"

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"prompt":            benignPrompt,
			"blocked":           blocked,
			"selected_decision": selectedDecision,
			"status_code":       response.StatusCode,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] blocked=%v selected_decision=%q status=%d\n", blocked, selectedDecision, response.StatusCode)
	}

	if !blocked {
		return fmt.Errorf(
			"expected the request to be blocked (fail-closed jailbreak signal against an unreachable deployment), "+
				"but x-vsr-response-path=%q selected_decision=%q status=%d",
			responsePath, selectedDecision, response.StatusCode,
		)
	}
	if selectedDecision != "block_on_classifier_error" {
		return fmt.Errorf(
			"request was blocked by decision %q, want %q - explicit no_match or legacy fail-closed behavior may be broken",
			selectedDecision, "block_on_classifier_error",
		)
	}

	return nil
}
