package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetLanguageDecision is the decision configured to route on the
// language signal matching Spanish (see e2e/profiles/ai-gateway/values.yaml).
const targetLanguageDecision = "spanish_language"

func init() {
	pkgtestcases.Register("language-routing", pkgtestcases.TestCase{
		Description: "Test language signal rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "language"},
		Fn:          testLanguageRouting,
	})
}

func testLanguageRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/language_routing_cases.json",
		MatchedHeader:  "x-vsr-matched-language",
		TargetDecision: targetLanguageDecision,
		ResultsTitle:   "LANGUAGE ROUTING TEST RESULTS",
		LogLabel:       "Language",
	})
}
