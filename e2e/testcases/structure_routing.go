package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetStructureDecision is one of the decisions configured to route on the
// request shape: keyword rules that count questions or find numbered steps
// (see the structure recipe in e2e/profiles/signal-routing/values.yaml). It is used only to
// confirm the negative case doesn't accidentally land on it; no keyword
// signal in the trace is what actually proves neither rule matched.
const targetStructureDecision = "many_questions_routing"

func init() {
	pkgtestcases.Register("structure-routing", pkgtestcases.TestCase{
		Description: "Test keyword count routing on request shape",
		Tags:        []string{"kubernetes", "routing", "keyword"},
		Fn:          testStructureRouting,
	})
}

func testStructureRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/structure_routing_cases.json",
		SignalType:     "keyword",
		TargetDecision: targetStructureDecision,
		ResultsTitle:   "STRUCTURE ROUTING TEST RESULTS",
		LogLabel:       "Structure",
		Entrypoint:     "vllm-sr/structure",
	})
}
