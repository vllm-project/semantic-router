package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetPreferenceExternalDecision is one of the decisions configured to
// route on the preference signal (see
// e2e/profiles/preference-external-routing/values.yaml). It is used only to
// confirm the negative case doesn't accidentally land on it; the empty
// x-vsr-matched-preference header is what actually proves no rule matched.
const targetPreferenceExternalDecision = "terse_preference_routing"

func init() {
	pkgtestcases.Register("preference-external-routing", pkgtestcases.TestCase{
		Description: "Test preference signal external (LLM-backed) rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "preference"},
		Fn:          testPreferenceExternalRouting,
	})
}

func testPreferenceExternalRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/preference_external_routing_cases.json",
		MatchedHeader:  "x-vsr-matched-preference",
		TargetDecision: targetPreferenceExternalDecision,
		ResultsTitle:   "PREFERENCE EXTERNAL ROUTING TEST RESULTS",
		LogLabel:       "Preference External",
	})
}
