package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetComplexityDecision is the decision the complexity signal escalates to
// (see e2e/profiles/complexity-routing/values.yaml). Complexity always emits a
// verdict, so no case expects an empty matched header: every case pins the
// exact "<rule>:<difficulty>" it should produce.
const targetComplexityDecision = "complexity_hard_routing"

func init() {
	pkgtestcases.Register("complexity-routing", pkgtestcases.TestCase{
		Description: "Test complexity signal local prototype scoring and routing",
		Tags:        []string{"kubernetes", "routing", "complexity"},
		Fn:          testComplexityRouting,
	})
}

func testComplexityRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/complexity_routing_cases.json",
		MatchedHeader:  "x-vsr-matched-complexity",
		TargetDecision: targetComplexityDecision,
		ResultsTitle:   "COMPLEXITY ROUTING TEST RESULTS",
		LogLabel:       "Complexity",
	})
}
