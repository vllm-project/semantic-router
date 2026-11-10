package extproc

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type modelCardProvider interface {
	Card(context.Context, string) (modelservice.ModelCard, error)
}

// prepareDecisionSelectionCards admits the System One questions decisions ask
// about their candidates while building a generation: the systemone
// selector's, a Chat escalation's answer_check or refusal, and the judges of
// match blocks. Requests only read this snapshot; no discovery runs on their
// selection path.
func prepareDecisionSelectionCards(cfg *config.RouterConfig, provider modelCardProvider) (map[string]modelservice.ModelCard, error) {
	cards := map[string]modelservice.ModelCard{}
	if cfg == nil {
		return cards, nil
	}
	selectionTask, _ := modelservice.BuiltinTask("model_selection")
	for _, recipe := range cfg.ReachableRoutingRecipes() {
		for _, decision := range recipe.Profile.Decisions {
			algorithm := decision.Algorithm
			switch {
			case algorithm != nil && algorithm.Type == config.DecisionAlgorithmSystemOne && algorithm.SystemOne != nil:
				selector := *algorithm.SystemOne
				deployment := cfg.SystemOneSelectorDeployment(selector)
				card, err := preparedCard(cards, cfg, provider, deployment, selector.EffectiveTimeout())
				if err != nil {
					return nil, fmt.Errorf("prepare decision selector %s: %w", decision.Name, err)
				}
				question := modelservice.Question{ID: decisionSelectorQuestionID, Type: config.SystemOneQuestionChoice, Instructions: selector.Instructions, Truncate: true}
				for _, candidate := range decision.ModelRefs {
					question.Choices = append(question.Choices, modelservice.Choice{Key: candidate.Model, Description: selector.Candidates[candidate.Model]})
				}
				if _, err := modelservice.CompileTask(selectionTask, question, card); err != nil {
					return nil, fmt.Errorf("prepare decision selector %s: %w", decision.Name, err)
				}
			case algorithm.EscalateChecksAnswers():
				if err := prepareEscalateCheck(cards, cfg, provider, algorithm.Escalate.Accept); err != nil {
					return nil, fmt.Errorf("prepare escalate %s: %w", decision.Name, err)
				}
			}
		}
	}
	for _, judge := range matchJudges(cfg) {
		deployment := cfg.MatchJudgeDeployment(judge)
		card, err := preparedCard(cards, cfg, provider, deployment, judge.EffectiveTimeout())
		if err == nil {
			err = match.Prepare(judge, card)
		}
		if err != nil {
			return nil, fmt.Errorf("prepare match judge on deployment %s: %w", deployment, err)
		}
	}
	return cards, nil
}

// prepareEscalateCheck compiles a Chat escalation's answer check on the card
// of the deployment that answers it.
func prepareEscalateCheck(cards map[string]modelservice.ModelCard, cfg *config.RouterConfig, provider modelCardProvider, accept *config.EscalateAccept) error {
	task := "refusal"
	check := config.EscalateAnswerCheck{}
	if accept.SystemOne != nil {
		task, check = "answer_check", *accept.SystemOne
	}
	card, err := preparedCard(cards, cfg, provider, cfg.EscalateCheckDeployment(accept), check.EffectiveTimeout())
	if err != nil {
		return err
	}
	definition, _ := modelservice.BuiltinTask(task)
	_, err = modelservice.CompileTask(definition, definition.Question, card)
	return err
}

// preparedCard is the card of a deployment: a systemone deployment's
// declared one, or the model runtime's, read once per generation.
func preparedCard(cards map[string]modelservice.ModelCard, cfg *config.RouterConfig, provider modelCardProvider, deployment string, timeout time.Duration) (modelservice.ModelCard, error) {
	if card, exists := cards[deployment]; exists {
		return card, nil
	}
	if remote := cfg.ModelDeployments[deployment]; remote.IsSystemOne() {
		card := modelservice.RemoteCard(remote.Model, remote.DeclaredQuestionTypes(), remote.DeclaredPresets())
		cards[deployment] = card
		return card, nil
	}
	if provider == nil {
		return modelservice.ModelCard{}, fmt.Errorf("deployment %s requires a model runtime", deployment)
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	card, err := provider.Card(ctx, deployment)
	cancel()
	if err != nil {
		return modelservice.ModelCard{}, err
	}
	if !card.Serves("decisions") {
		return modelservice.ModelCard{}, fmt.Errorf("deployment %s requires the decisions surface", deployment)
	}
	cards[deployment] = card
	return card, nil
}
