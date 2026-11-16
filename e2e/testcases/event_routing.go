package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetEventDecision is the decision configured to route on the
// critical_payment_event keyword rule, whose keywords are the event types,
// severity and action code of a payment incident (see
// e2e/profiles/ai-gateway/values.yaml).
const targetEventDecision = "critical_event"

func init() {
	pkgtestcases.Register("event-routing", pkgtestcases.TestCase{
		Description: "Test keyword routing of event payloads",
		Tags:        []string{"kubernetes", "routing", "keyword"},
		Fn:          testEventRouting,
	})
}

func testEventRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/event_routing_cases.json",
		SignalType:     "keyword",
		TargetDecision: targetEventDecision,
		ResultsTitle:   "EVENT ROUTING TEST RESULTS",
		LogLabel:       "Event",
	})
}
