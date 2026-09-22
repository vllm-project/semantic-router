package testcases

import (
	"context"
	"fmt"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("decision-scoped-random", pkgtestcases.TestCase{
		Description: "Verify a random decision selects one of its declared candidates",
		Tags:        []string{"routing", "selection", "random", "regression"},
		Fn:          testDecisionScopedRandom,
	})
}

// Checks wiring and eligibility only; uniformity is covered by the selector unit tests (#3273).
func testDecisionScopedRandom(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	const (
		query        = "Apply the decision scoped random policy"
		wantDecision = "random_policy_decision"
		wantMethod   = "random"
	)
	candidates := map[string]bool{"premium-model": true, "economy-model": true}

	header, err := requestDecisionScopedHeaders(ctx, localPort, query)
	if err != nil {
		return err
	}
	decision := header.Get("x-vsr-selected-decision")
	method := header.Get("x-vsr-selected-algorithm")
	model := header.Get("x-vsr-selected-model")

	if decision != wantDecision || method != wantMethod || !candidates[model] {
		return fmt.Errorf(
			"query %q selected decision=%q algorithm=%q model=%q, want decision=%q algorithm=%q model in %v",
			query, decision, method, model, wantDecision, wantMethod, candidates,
		)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"decision":       decision,
			"algorithm":      method,
			"selected_model": model,
		})
	}
	return nil
}
