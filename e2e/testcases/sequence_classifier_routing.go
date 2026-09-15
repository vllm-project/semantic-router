package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const sequenceClassifierDecision = "sequence_classifier_toxic"

func init() {
	pkgtestcases.Register("sequence-classifier-routing", pkgtestcases.TestCase{
		Description: "Verify outbound http_classify scores drive a generic classifier decision",
		Tags:        []string{"classifier", "http-classify", "routing"},
		Fn:          testSequenceClassifierRouting,
	})
}

func testSequenceClassifierRouting(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	cases := []struct {
		name      string
		prompt    string
		wantMatch bool
	}{
		{
			name:      "high toxic score matches",
			prompt:    "__SEQUENCE_CLASSIFIER_TOXIC__ verify the remote classifier path",
			wantMatch: true,
		},
		{
			name:      "low toxic score does not match",
			prompt:    "__SEQUENCE_CLASSIFIER_BENIGN__ verify the remote classifier path",
			wantMatch: false,
		},
	}

	decisions := make(map[string]string, len(cases))
	for _, testCase := range cases {
		response, err := sendLocalChatCompletion(ctx, localPort, "auto", testCase.prompt, 30*time.Second)
		if err != nil {
			return fmt.Errorf("%s: %w", testCase.name, err)
		}
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("%s: %s", testCase.name, formatUnexpectedChatCompletionStatus(response))
		}
		decision := response.Headers.Get("x-vsr-selected-decision")
		decisions[testCase.name] = decision
		matched := decision == sequenceClassifierDecision
		if matched != testCase.wantMatch {
			return fmt.Errorf(
				"%s: x-vsr-selected-decision=%q, expected sequence classifier match=%t",
				testCase.name,
				decision,
				testCase.wantMatch,
			)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"routing_cases":  len(cases),
			"routing_passed": len(cases),
			"decisions":      decisions,
		})
	}
	return nil
}
