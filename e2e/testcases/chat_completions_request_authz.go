package testcases

import (
	"context"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

var authzChatRequestHeaders = map[string]string{
	"Authorization": "Bearer " + authzJWTChatToken,
}

func init() {
	pkgtestcases.Register("chat-completions-request-authz", pkgtestcases.TestCase{
		Description: "Send a chat completions request with a valid authz JWT and verify 200 OK response",
		Tags:        []string{"llm", "functional", "authz"},
		Fn:          testChatCompletionsRequestAuthz,
	})
}

func testChatCompletionsRequestAuthz(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	return runChatCompletionsRequest(ctx, client, opts, authzChatRequestHeaders)
}
