package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

// These names match the agentic-facts-policy recipe in
// e2e/profiles/routing-strategies/values.yaml and the router's default
// carrier and trust-marker headers.
const (
	agenticFactsPolicyModel   = "vllm-sr/agentic-facts-policy"
	agenticFactsCarrierHeader = "x-vsr-agentic-facts"
	agenticFactsTrustHeader   = "x-vsr-agentic-facts-trusted"
	agenticFactsTrustValue    = "1"
	agenticReviewerDecision   = "agentic_reviewer_decision"
	agenticDefaultDecision    = "agentic_default_decision"
)

// agenticFactsEnvelope returns a valid reviewer envelope as a map, so each
// case can change one field. It expires two minutes from now, inside the
// router's default five-minute maximum lifetime.
func agenticFactsEnvelope() map[string]any {
	return map[string]any{
		"version":        "1",
		"delegated_role": "reviewer",
		"expires_at":     time.Now().Add(2 * time.Minute).UTC().Format(time.RFC3339),
	}
}

func encodeAgenticFactsEnvelope(envelope map[string]any) (string, error) {
	raw, err := json.Marshal(envelope)
	if err != nil {
		return "", fmt.Errorf("encode agentic facts envelope: %w", err)
	}
	return string(raw), nil
}

// agenticFactsRoute is what one request reveals about routing.
type agenticFactsRoute struct {
	StatusCode int
	Decision   string
	Model      string
	Body       []byte
}

// sendAgenticFactsRequest plays the trusted gateway: when trusted is true it
// sets the trust marker itself, which a real deployment's gateway would do
// after authenticating the caller. An empty carrier sends no envelope.
func sendAgenticFactsRequest(
	ctx context.Context,
	chat *fixtures.ChatCompletionsClient,
	carrier string,
	trusted bool,
) (agenticFactsRoute, error) {
	headers := map[string]string{}
	if carrier != "" {
		headers[agenticFactsCarrierHeader] = carrier
	}
	if trusted {
		headers[agenticFactsTrustHeader] = agenticFactsTrustValue
	}

	response, err := chat.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: agenticFactsPolicyModel,
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "review this change"},
		},
	}, headers)
	if err != nil {
		return agenticFactsRoute{}, err
	}
	return agenticFactsRoute{
		StatusCode: response.StatusCode,
		Decision:   response.Headers.Get("x-vsr-selected-decision"),
		Model:      response.Headers.Get("x-vsr-selected-model"),
		Body:       response.Body,
	}, nil
}
