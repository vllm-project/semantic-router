package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetContextDecision is the decision configured to route on the context
// signal's long_context rule (see e2e/profiles/ai-gateway/values.yaml).
const targetContextDecision = "long_context_routing"

func init() {
	pkgtestcases.Register("context-routing", pkgtestcases.TestCase{
		Description: "Test context signal rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "context"},
		Fn:          testContextRouting,
	})
}

func testContextRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/context_routing_cases.json",
		MatchedHeader:  "x-vsr-matched-context",
		TargetDecision: targetContextDecision,
		ResultsTitle:   "CONTEXT ROUTING TEST RESULTS",
		LogLabel:       "Context",
	})
}
