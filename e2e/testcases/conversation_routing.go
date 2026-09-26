package testcases

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// targetConversationDecision is one of the decisions configured to route on
// the conversation signal (see e2e/profiles/conversation-routing/values.yaml).
const targetConversationDecision = "multi_turn_user_routing"

func init() {
	pkgtestcases.Register("conversation-routing", pkgtestcases.TestCase{
		Description: "Test conversation signal rule matching and routing",
		Tags:        []string{"kubernetes", "routing", "conversation"},
		Fn:          testConversationRouting,
	})
}

func testConversationRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runSignalRoutingTest(ctx, client, opts, signalRoutingConfig{
		TestDataPath:   "e2e/testcases/testdata/conversation_routing_cases.json",
		MatchedHeader:  "x-vsr-matched-conversation",
		TargetDecision: targetConversationDecision,
		ResultsTitle:   "CONVERSATION ROUTING TEST RESULTS",
		LogLabel:       "Conversation",
	})
}
