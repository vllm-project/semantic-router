package testcases

import (
	"context"
	"fmt"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("shadow-dispatch-group-cost-budget", pkgtestcases.TestCase{
		Description: "A Shadow group at the cost limit dispatches both candidates; an over-budget group dispatches neither and preserves the primary response",
		Tags:        []string{"router-replay", "shadow-dispatch", "failure-isolation"},
		Priority:    pkgtestcases.P0,
		Fn:          testShadowDispatchGroupCostBudget,
	})
}

func testShadowDispatchGroupCostBudget(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	run, err := openShadowDispatchRun(ctx, client, opts)
	if err != nil {
		return err
	}
	defer run.Close()

	arms := []string{"budget-a", "budget-b"}
	for _, arm := range arms {
		if err := run.control(ctx, arm, "reset", "healthy"); err != nil {
			return err
		}
	}

	// At 100 output tokens the group costs 0.30 USD. With 101, each
	// candidate still fits individually, but their combined cost is 0.303 USD.
	for _, tc := range []struct {
		outputLimit int
		reason      string
		callsPerArm int
	}{
		{101, "budget_exhausted", 0},
		{100, "completed", 1},
	} {
		session, err := run.primaryWithOutputLimit(ctx, "vllm-sr/shadow-cost-budget", shadowPrimaryCeiling, tc.outputLimit)
		if err != nil {
			return fmt.Errorf("primary request with output limit %d: %w", tc.outputLimit, err)
		}
		record, err := run.replay(ctx, session, 2)
		if err != nil {
			return fmt.Errorf("replay for output limit %d: %w", tc.outputLimit, err)
		}
		reasons := make(map[string]string, len(arms))
		for _, arm := range arms {
			reasons["openai/shadow-"+arm] = tc.reason
		}
		if err := requireShadowMultiOutcomes(record, reasons); err != nil {
			return fmt.Errorf("shadow outcomes for output limit %d: %w", tc.outputLimit, err)
		}
		if tc.callsPerArm == 1 {
			wantCost := map[string]string{"openai/shadow-budget-a": "0.1", "openai/shadow-budget-b": "0.2"}
			for _, outcome := range record.Outcomes {
				cost := outcome.Metadata["budget_estimated_cost"]
				currency := outcome.Metadata["budget_currency"]
				if cost != wantCost[outcome.TargetRef] || currency != "USD" {
					return fmt.Errorf("replay %s shadow %s estimated cost = %q %s, want %s USD",
						record.ID, outcome.TargetRef, cost, currency, wantCost[outcome.TargetRef])
				}
			}
		}
		for _, arm := range arms {
			if _, err := run.waitProvider(ctx, arm, tc.callsPerArm, 0); err != nil {
				return fmt.Errorf("cost budget with output limit %d: %w", tc.outputLimit, err)
			}
		}
	}
	return nil
}
