package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetReaskDecision is the decision configured to route on the
// repeated_billing_question reask rule (see e2e/profiles/ai-gateway/values.yaml).
const targetReaskDecision = "reask_escalation"

func init() {
	pkgtestcases.Register("reask-routing", pkgtestcases.TestCase{
		Description: "Test reask signal rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "reask"},
		Fn:          testReaskRouting,
	})
}

func testReaskRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/reask_routing_cases.json",
		MatchedHeader:  "x-vsr-matched-reask",
		TargetDecision: targetReaskDecision,
		ResultsTitle:   "REASK ROUTING TEST RESULTS",
		LogLabel:       "Reask",
	})
}
