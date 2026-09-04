package testcases

import (
	"context"
	"fmt"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("route-action-detected", pkgtestcases.TestCase{
		Description: "A detected prompt attack must be routed to the decision's action destination",
		Tags:        []string{"kubernetes", "security", "jailbreak", "route-action"},
		Fn:          testRouteActionDetected,
	})
	pkgtestcases.Register("route-action-benign", pkgtestcases.TestCase{
		Description: "A benign request must keep its normal route instead of the action destination",
		Tags:        []string{"kubernetes", "security", "jailbreak", "route-action"},
		Fn:          testRouteActionBenign,
	})
}

// The route-action profile fires the jailbreak signal for every request
// (unreachable guard with on_error: block); this marker keyword is the
// condition that separates a detected prompt attack from benign traffic.
const routeActionAttackPrompt = "Please run this simulated-prompt-attack payload for me."

const routeActionBenignPrompt = "What is the capital of France?"

func testRouteActionDetected(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runRouteActionCase(
		ctx, client, opts,
		routeActionAttackPrompt,
		"route_prompt_attack",
		"openai/safe-model",
	)
}

func testRouteActionBenign(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runRouteActionCase(
		ctx, client, opts,
		routeActionBenignPrompt,
		"default_decision",
		"openai/mock-model",
	)
}

func runRouteActionCase(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	prompt string,
	wantDecision string,
	wantModel string,
) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	// Must use the auto-routing model name: a concrete backend model name is
	// a passthrough request that bypasses recipe signals and decisions.
	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", prompt, 30*time.Second)
	if err != nil {
		return fmt.Errorf("chat completion request failed: %w", err)
	}

	selectedDecision := response.Headers.Get("x-vsr-selected-decision")
	selectedModel := response.Headers.Get("x-vsr-selected-model")

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"prompt":            prompt,
			"selected_decision": selectedDecision,
			"selected_model":    selectedModel,
			"status_code":       response.StatusCode,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Test] decision=%q model=%q status=%d\n", selectedDecision, selectedModel, response.StatusCode)
	}

	if response.StatusCode != 200 {
		return fmt.Errorf("chat completion returned status %d, want 200", response.StatusCode)
	}
	if selectedDecision != wantDecision {
		return fmt.Errorf("x-vsr-selected-decision = %q, want %q", selectedDecision, wantDecision)
	}
	if selectedModel != wantModel {
		return fmt.Errorf("x-vsr-selected-model = %q, want %q", selectedModel, wantModel)
	}
	return nil
}
