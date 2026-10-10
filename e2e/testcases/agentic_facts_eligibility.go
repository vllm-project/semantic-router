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
	pkgtestcases.Register("agentic-facts-eligibility", pkgtestcases.TestCase{
		Description: "Required capabilities narrow the decision's models and fail closed when none remain",
		Tags:        []string{"signal-decision", "agentic-facts", "model-selection", "security"},
		Fn:          testAgenticFactsEligibility,
	})
}

// The recipe opts into strict candidate requirements, and its model cards
// declare real protocol capabilities: the preferred model [text], the
// alternate [text, tools]. No model declares reasoning.
const (
	agenticPreferredModel      = "agentic-preferred-model"
	agenticAlternateModel      = "agentic-alternate-model"
	agenticNoModelCapability   = "reasoning"
	agenticAlternateCapability = "tools"
	// agenticNoEligibleModelMessage is the router's fixed client message when
	// strict candidate requirements leave no model.
	agenticNoEligibleModelMessage = "no model is eligible to serve the request"
)

type agenticFactsEligibilityCase struct {
	name       string
	requires   []string
	wantStatus int
	wantModel  string // checked only when wantStatus is 200
}

// agenticFactsEligibilityCases all send a trusted, valid reviewer envelope, so
// every case reaches agentic_reviewer_decision. Only required_capabilities
// changes. The first case is the control: without it, the second could pass
// even if the router always picked the alternate model.
func agenticFactsEligibilityCases() []agenticFactsEligibilityCase {
	return []agenticFactsEligibilityCase{
		{
			name:       "no required capability keeps the quality preference",
			wantStatus: http.StatusOK,
			wantModel:  agenticPreferredModel,
		},
		{
			name:       "required capability removes the preferred model",
			requires:   []string{agenticAlternateCapability},
			wantStatus: http.StatusOK,
			wantModel:  agenticAlternateModel,
		},
		{
			name:       "capability no model has fails closed",
			requires:   []string{agenticNoModelCapability},
			wantStatus: http.StatusServiceUnavailable,
		},
	}
}

func agenticFactsCarrierRequiring(capabilities []string) (string, error) {
	envelope := agenticFactsEnvelope()
	if len(capabilities) > 0 {
		envelope["required_capabilities"] = capabilities
	}
	return encodeAgenticFactsEnvelope(envelope)
}

func testAgenticFactsEligibility(
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

	cases := agenticFactsEligibilityCases()
	var failures []string
	results := map[string]interface{}{}
	for _, tc := range cases {
		carrier, err := agenticFactsCarrierRequiring(tc.requires)
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		route, err := sendAgenticFactsRequest(ctx, chat, carrier, true)
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		results[tc.name] = map[string]interface{}{
			"status": route.StatusCode,
			"model":  route.Model,
		}
		if problem := checkAgenticFactsEligibility(tc, route); problem != "" {
			failures = append(failures, tc.name+": "+problem)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(results)
	}
	if len(failures) > 0 {
		return fmt.Errorf("agentic facts eligibility: %d of %d cases failed:\n%s",
			len(failures), len(cases), strings.Join(failures, "\n"))
	}
	return nil
}

func checkAgenticFactsEligibility(tc agenticFactsEligibilityCase, route agenticFactsRoute) string {
	if route.StatusCode != tc.wantStatus {
		return fmt.Sprintf("HTTP %d, want %d: %s", route.StatusCode, tc.wantStatus, route.Body)
	}
	if tc.wantStatus != http.StatusOK {
		return checkAgenticFactsFailClosedBody(route.Body)
	}
	if route.Decision != agenticReviewerDecision {
		return fmt.Sprintf("decision = %q, want %q", route.Decision, agenticReviewerDecision)
	}
	if route.Model != tc.wantModel {
		return fmt.Sprintf("model = %q, want %q", route.Model, tc.wantModel)
	}
	return ""
}

// checkAgenticFactsFailClosedBody checks the 503 body carries the router's
// fixed message and nothing else. The body goes back to the caller, so it must
// not reveal the operator's model names, and it carries no value the caller
// sent.
func checkAgenticFactsFailClosedBody(body []byte) string {
	text := string(body)
	if !strings.Contains(text, agenticNoEligibleModelMessage) {
		return fmt.Sprintf("503 body does not carry the no-eligible-model message: %s", text)
	}
	for _, private := range []string{agenticPreferredModel, agenticAlternateModel, agenticNoModelCapability} {
		if strings.Contains(text, private) {
			return fmt.Sprintf("503 body contains %q: %s", private, text)
		}
	}
	return ""
}
