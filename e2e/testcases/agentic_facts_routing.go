package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("agentic-facts-routing", pkgtestcases.TestCase{
		Description: "Agentic facts influence routing only when the envelope is trusted and valid",
		Tags:        []string{"signal-decision", "agentic-facts", "routing", "security"},
		Fn:          testAgenticFactsRouting,
	})
}

type agenticFactsRoutingCase struct {
	name string
	// carrier returns the raw header value. Most cases change one field of a
	// valid envelope; the malformed case returns text that is not JSON.
	carrier      func() (string, error)
	trusted      bool
	wantDecision string
}

func agenticFactsCarrierWith(change func(envelope map[string]any)) func() (string, error) {
	return func() (string, error) {
		envelope := agenticFactsEnvelope()
		change(envelope)
		return encodeAgenticFactsEnvelope(envelope)
	}
}

func agenticFactsRoutingCases() []agenticFactsRoutingCase {
	unchanged := func(map[string]any) {}
	return []agenticFactsRoutingCase{
		{
			name:         "no envelope",
			carrier:      func() (string, error) { return "", nil },
			trusted:      false,
			wantDecision: agenticDefaultDecision,
		},
		{
			name:         "authenticated valid envelope",
			carrier:      agenticFactsCarrierWith(unchanged),
			trusted:      true,
			wantDecision: agenticReviewerDecision,
		},
		{
			name:         "untrusted envelope is ignored",
			carrier:      agenticFactsCarrierWith(unchanged),
			trusted:      false,
			wantDecision: agenticDefaultDecision,
		},
		{
			name:         "malformed envelope",
			carrier:      func() (string, error) { return "not json at all", nil },
			trusted:      true,
			wantDecision: agenticDefaultDecision,
		},
		{
			name: "stale envelope",
			carrier: agenticFactsCarrierWith(func(envelope map[string]any) {
				envelope["expires_at"] = time.Now().Add(-10 * time.Minute).UTC().Format(time.RFC3339)
			}),
			trusted:      true,
			wantDecision: agenticDefaultDecision,
		},
		{
			name: "nested delegation within the depth bound",
			carrier: agenticFactsCarrierWith(func(envelope map[string]any) {
				envelope["lineage"] = map[string]any{
					"root_invocation_id":   "inv-root",
					"parent_invocation_id": "inv-parent",
					"depth":                3,
				}
			}),
			trusted:      true,
			wantDecision: agenticReviewerDecision,
		},
		{
			name: "nested delegation deeper than the bound",
			carrier: agenticFactsCarrierWith(func(envelope map[string]any) {
				envelope["lineage"] = map[string]any{
					"root_invocation_id":   "inv-root",
					"parent_invocation_id": "inv-parent",
					"depth":                17,
				}
			}),
			trusted:      true,
			wantDecision: agenticDefaultDecision,
		},
		{
			name: "conflicting lineage",
			carrier: agenticFactsCarrierWith(func(envelope map[string]any) {
				envelope["lineage"] = map[string]any{
					"root_invocation_id": "inv-root",
					"depth":              2,
				}
			}),
			trusted:      true,
			wantDecision: agenticDefaultDecision,
		},
	}
}

func testAgenticFactsRouting(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open session: %w", err)
	}
	defer session.Close()
	chat := fixtures.NewChatCompletionsClient(session, 45*time.Second)

	cases := agenticFactsRoutingCases()
	var failures []string
	results := map[string]interface{}{}
	for _, tc := range cases {
		carrier, err := tc.carrier()
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		route, err := sendAgenticFactsRequest(ctx, chat, carrier, tc.trusted)
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		results[tc.name] = route.Decision
		switch {
		case route.StatusCode != http.StatusOK:
			failures = append(failures, fmt.Sprintf(
				"%s: HTTP %d: %s", tc.name, route.StatusCode, route.Body))
		case route.Decision != tc.wantDecision:
			failures = append(failures, fmt.Sprintf(
				"%s: decision = %q, want %q", tc.name, route.Decision, tc.wantDecision))
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(results)
	}
	if len(failures) > 0 {
		return fmt.Errorf("agentic facts routing: %d of %d cases failed:\n%s",
			len(failures), len(cases), strings.Join(failures, "\n"))
	}
	return nil
}
