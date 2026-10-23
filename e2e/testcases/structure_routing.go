package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetStructureDecision is one of the decisions configured to route on the
// structure signal (see e2e/profiles/structure-routing/values.yaml). It is
// used only to confirm the negative case doesn't accidentally land on it;
// the empty x-vsr-matched-structure header is what actually proves neither
// structure rule matched.
const targetStructureDecision = "many_questions_routing"

func init() {
	pkgtestcases.Register("structure-routing", pkgtestcases.TestCase{
		Description: "Test structure signal rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "structure"},
		Fn:          testStructureRouting,
	})
}

func testStructureRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/structure_routing_cases.json",
		MatchedHeader:  "x-vsr-matched-structure",
		TargetDecision: targetStructureDecision,
		ResultsTitle:   "STRUCTURE ROUTING TEST RESULTS",
		LogLabel:       "Structure",
	})
}
