package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetPreferenceDecision is one of the decisions configured to route on the
// preference rule's labels (see e2e/profiles/preference-routing/values.yaml).
// The rule is a systemone choice, so every case reports the option the System
// One fixture chose in the trace's systemone signals, none included.
const targetPreferenceDecision = "terse_preference_routing"

func init() {
	pkgtestcases.Register("preference-routing", pkgtestcases.TestCase{
		Description: "Test preference rule label matching and routing",
		Tags:        []string{"kubernetes", "routing", "preference"},
		Fn:          testPreferenceRouting,
	})
}

func testPreferenceRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/preference_routing_cases.json",
		SignalType:     "systemone",
		TargetDecision: targetPreferenceDecision,
		ResultsTitle:   "PREFERENCE ROUTING TEST RESULTS",
		LogLabel:       "Preference",
	})
}
