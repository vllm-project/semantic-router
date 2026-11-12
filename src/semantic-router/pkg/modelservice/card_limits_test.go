package modelservice

import (
	"context"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

func TestACardReportsTheLengthItsModelWasValidatedAt(t *testing.T) {
	maxInput, validated := 262_144, 32_768
	card := decodeCard(api.ModelCard{Id: "d3", Surfaces: []string{"decisions"}, Limits: &api.ModelLimits{MaxInputTokens: &maxInput, ValidatedInputTokens: &validated}})
	if card.QuestionTokens() != validated {
		t.Fatalf("a validated card caps a question at %d, got %d", validated, card.QuestionTokens())
	}
	card = decodeCard(api.ModelCard{Id: "kai", Surfaces: []string{"decisions"}, Limits: &api.ModelLimits{MaxInputTokens: &validated}})
	if card.QuestionTokens() != validated {
		t.Fatalf("a card without a validated length caps a question at its maximum, got %d", card.QuestionTokens())
	}
}

func TestAListedCardGivesItsInputLimits(t *testing.T) {
	const listing = `{"models":[{"name":"vllm-sr/d3-ultra","description":"","release_date":"","version":"5.0","card":{"question_types":["choice","noul"],"limits":%s}}]}`
	for name, tc := range map[string]struct {
		limits              string
		maxInput, validated int
	}{
		"both":      {`{"max_input_tokens":262144,"validated_input_tokens":32768}`, 262_144, 32_768},
		"maximum":   {`{"max_input_tokens":32768}`, 32_768, 0},
		"malformed": {`{"max_input_tokens":"many","validated_input_tokens":1.5}`, 0, 0},
		"none":      {`null`, 0, 0},
	} {
		t.Run(name, func(t *testing.T) {
			remote := fakeRemote(t, "hosted", &fakeSystemOneAPI{listing: fmt.Sprintf(listing, tc.limits)}, config.ModelDeployment{Schema: "vllm-sr/5.0"})
			card, err := remote.PrepareCard(context.Background(), "hosted")
			if err != nil || !card.Answers("noul") || card.MaxInputTokens != tc.maxInput || card.ValidatedInputTokens != tc.validated {
				t.Fatalf("card %+v, %v; want limits %d and %d", card, err, tc.maxInput, tc.validated)
			}
		})
	}
}
