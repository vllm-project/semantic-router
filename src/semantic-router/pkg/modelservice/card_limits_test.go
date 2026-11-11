package modelservice

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
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

// The states a stage asks one model about go in its batch in their requests'
// order, so the windows of a scan are read tail first.
func TestAStagesStatesGoInTheirRequestsOrder(t *testing.T) {
	runtime := runtimetest.New(runtimetest.Model{ID: "d3"})
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"d3"}})
	states := []struct {
		text  string
		order int
	}{{"window b, read third", 3}, {"a routing view, read first", 0}, {"window z, read second", 1}, {"window a, read fourth", 4}}
	askers := make([]asker, len(states))
	for index, state := range states {
		askers[index] = asker{deployments: []string{"d3"}, ask: func(ctx context.Context) error {
			_, err := lease.Decide(ctx, "d3", Request{State: state.text, Order: state.order, Questions: []Question{choiceQuestion("guard")}})
			return err
		}}
	}
	if _, errs := runStage(DefaultBundleWindow, askers...); errors.Join(errs...) != nil {
		t.Fatal(errors.Join(errs...))
	}
	decisions := runtime.Decisions()
	if len(decisions) != 1 || decisions[0].States == nil {
		t.Fatalf("decisions %+v, want one batch", decisions)
	}
	batch := *decisions[0].States
	for name, want := range map[string]string{"1": "a routing view, read first", "2": "window z, read second", "3": "window b, read third", "4": "window a, read fourth"} {
		if got := fmt.Sprint(batch[name].State); got != want {
			t.Fatalf("entry %s is %q, want %q", name, got, want)
		}
	}
}
