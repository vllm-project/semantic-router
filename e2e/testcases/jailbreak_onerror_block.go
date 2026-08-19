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
		Description: "Test that PromptGuardConfig.OnError: block closes the request when the guardrail classifier is unreachable (#2918)",
		Tags:        []string{"kubernetes", "security", "jailbreak", "prompt-guard"},
		Fn:          testJailbreakOnErrorBlock,
	})
}

// testJailbreakOnErrorBlock sends an ordinary, non-adversarial prompt against
// the jailbreak-onerror profile (prompt_guard.protocol points at an endpoint
// nothing listens on, with on_error: block). It must be blocked: with the
// classifier unreachable, on_error: block treats the classify failure as a
// positive detection rather than letting a request through unchecked.
func testJailbreakOnErrorBlock(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing PromptGuardConfig.OnError: block against an unreachable classifier")
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
	// signal or on_error: block being tested here.
	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", benignPrompt, 30*time.Second)
	if err != nil {
		return fmt.Errorf("chat completion request failed: %w", err)
	}

	fastResponse := response.Headers.Get("x-vsr-fast-response")
	selectedDecision := response.Headers.Get("x-vsr-selected-decision")
	blocked := fastResponse == "true"

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
			"expected the request to be blocked (on_error: block against an unreachable classifier), "+
				"but x-vsr-fast-response=%q selected_decision=%q status=%d",
			fastResponse, selectedDecision, response.StatusCode,
		)
	}
	if selectedDecision != "block_on_classifier_error" {
		return fmt.Errorf(
			"request was blocked, but by decision %q, want %q - the classify failure may not be reaching on_error: block",
			selectedDecision, "block_on_classifier_error",
		)
	}

	return nil
}
