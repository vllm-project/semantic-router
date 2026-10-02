package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// The action decisions live in the e2e-action recipe of
// e2e/profiles/ai-gateway/values.yaml, so they never compete with the default
// recipe's domain routes. targetActionDecision routes explanation requests.
const (
	actionRoutingModel   = "e2e-action"
	targetActionDecision = "explain_action"
)

func init() {
	pkgtestcases.Register("action-routing", pkgtestcases.TestCase{
		Description: "Test action signal rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "action"},
		Fn:          testActionRouting,
	})
}

func testActionRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/action_routing_cases.json",
		MatchedHeader:  "x-vsr-matched-action",
		TargetDecision: targetActionDecision,
		ResultsTitle:   "ACTION ROUTING TEST RESULTS",
		LogLabel:       "Action",
		Model:          actionRoutingModel,
	})
}
