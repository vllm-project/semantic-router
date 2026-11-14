package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// The masking-rules recipe masks with no pii rule: a regex keyword rule
// finds employee IDs and a systemone span rule, asked of provider-mocker's
// System One fixture, finds account numbers.
const (
	maskingRulesEntrypoint = "vllm-sr/masking-rules"
	maskingRulesDecision   = "mask_rules"
	maskingRulesEmployee   = "E-104233"
	maskingRulesAccount    = "88-1910"
)

func init() {
	pkgtestcases.Register("masking-own-rules", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P0,
		Description: "What a regex keyword rule and a System One span rule find reaches the backend under each rule's placeholder, on Chat and Anthropic, and a span rule that fails answers 503",
		Tags:        []string{"masking", "keyword", "systemone", "functional", "fail-closed"},
		Fn:          testMaskingOwnRules,
	})
}

func testMaskingOwnRules(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	prompt := "Paid by " + maskingRulesEmployee + " for account " + maskingRulesAccount +
		" " + maskingSpanMarker + " account " + maskingRulesAccount
	want := "Paid by [EMPLOYEE_0] for account [CUSTOMER_ACCOUNT_0]"
	received := map[string]string{}
	for _, tc := range []maskingWireCase{
		{
			name: "chat completions", path: "/v1/chat/completions", decision: maskingRulesDecision,
			body: map[string]interface{}{
				"model":    maskingRulesEntrypoint,
				"messages": []map[string]interface{}{{"role": "user", "content": prompt}},
			},
		},
		{
			name: "anthropic messages", path: "/v1/messages", decision: maskingRulesDecision,
			body: map[string]interface{}{
				"model": maskingRulesEntrypoint, "max_tokens": 64,
				"messages": []map[string]interface{}{{"role": "user", "content": prompt}},
			},
		},
	} {
		echoed, caseErr := runMaskingWireCase(ctx, session, tc)
		if caseErr != nil {
			return fmt.Errorf("%s: %w", tc.name, caseErr)
		}
		if !strings.HasPrefix(echoed, want) || strings.Contains(echoed, maskingRulesEmployee) || strings.Contains(echoed, maskingRulesAccount) {
			return fmt.Errorf("%s: the backend received %q, want it to start %q with no employee ID or account number", tc.name, echoed, want)
		}
		received[tc.name] = echoed
	}

	failed, err := runMaskingRawRequest(ctx, session, "/v1/chat/completions", map[string]interface{}{
		"model":    maskingRulesEntrypoint,
		"messages": []map[string]interface{}{{"role": "user", "content": "Paid by " + maskingRulesEmployee + " " + maskingErrorMarker}},
	})
	if err != nil {
		return err
	}
	if failed.StatusCode != http.StatusServiceUnavailable || strings.Contains(string(failed.Body), maskingRulesEmployee) {
		return fmt.Errorf("a span rule that failed answered HTTP %d %q, want 503 without the request's text", failed.StatusCode, truncateString(string(failed.Body), 300))
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"received": received, "failed_status": failed.StatusCode})
	}
	return nil
}
