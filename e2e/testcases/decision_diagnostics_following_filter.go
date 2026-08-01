package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const decisionDiagnosticsTestHeader = "x-test-vsr-decision-diagnostics"

func init() {
	pkgtestcases.Register("decision-diagnostics-following-filter", pkgtestcases.TestCase{
		Description: "Verify a following Envoy filter traverses structured decision diagnostics",
		Tags:        []string{"routing", "extproc", "dynamic-metadata", "integration"},
		Fn:          testDecisionDiagnosticsFollowingFilter,
	})
}

func testDecisionDiagnosticsFollowingFilter(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	response, err := sendLocalChatCompletion(
		ctx,
		localPort,
		"MoM",
		"Apply the decision scoped quality policy",
		30*time.Second,
	)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("diagnostics request returned HTTP %d: %s", response.StatusCode, string(response.Body))
	}

	want := "quality_policy_decision:v1:quality_policy"
	if got := response.Headers.Get(decisionDiagnosticsTestHeader); got != want {
		return fmt.Errorf("following filter header %q = %q, want %q", decisionDiagnosticsTestHeader, got, want)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"following_filter_value": want})
	}
	return nil
}
