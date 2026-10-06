package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetPreferenceDecision is one of the decisions configured to route on the
// preference signal (see e2e/profiles/preference-routing/values.yaml). It is
// used only to confirm the negative case doesn't accidentally land on it;
// the empty x-vsr-matched-preference header is what actually proves neither
// preference rule matched.
const targetPreferenceDecision = "terse_preference_routing"

func init() {
	pkgtestcases.Register("preference-routing", pkgtestcases.TestCase{
		Description: "Test preference signal contrastive rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "preference"},
		Fn:          testPreferenceRouting,
	})
}

func testPreferenceRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/preference_routing_cases.json",
		MatchedHeader:  "x-vsr-matched-preference",
		TargetDecision: targetPreferenceDecision,
		ResultsTitle:   "PREFERENCE ROUTING TEST RESULTS",
		LogLabel:       "Preference",
	})
}
