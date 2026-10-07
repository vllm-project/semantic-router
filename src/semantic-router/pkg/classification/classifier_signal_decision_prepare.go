package classification

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// labelledDecisionRules are the used Set and Span rules: only models that
// declare those question types answer them, so preparation checks each one
// against its deployment's card.
func (c *Classifier) labelledDecisionRules() []config.DecisionSignalRule {
	if c == nil || c.Config == nil || !c.usesRoutingSignalType(config.SignalTypeDecision) {
		return nil
	}
	used := c.getUsedSignals()
	var rules []config.DecisionSignalRule
	for _, rule := range c.Config.DecisionRules {
		if (rule.Question.Type == config.DecisionQuestionSet || rule.Question.Type == config.DecisionQuestionSpan) &&
			signalRuleUsed(used, config.SignalTypeDecision, rule.Name) {
			rules = append(rules, rule)
		}
	}
	return rules
}

// prepareDecisionSignals fails the generation when a Set or Span question
// goes to a deployment whose model does not declare that question type: every
// request would otherwise be answered invalid_question and leave the signal
// unknown. Without model runtime services there is no card to check.
func (c *Classifier) prepareDecisionSignals() error {
	if c.models == nil || c.models.runtime == nil || c.models.runtime.Services() == nil {
		return nil
	}
	cards := make(map[string]modelservice.ModelCard)
	for _, rule := range c.labelledDecisionRules() {
		name, deployment := c.decisionRuleDeployment(rule)
		card, checked := cards[name]
		if !checked {
			var err error
			card, err = c.models.runtime.DeploymentCard(context.Background(), name, deployment)
			if err != nil {
				return fmt.Errorf("routing.signals.decision[%s]: %w", rule.Name, err)
			}
			cards[name] = card
		}
		if !card.Answers(rule.Question.Type) {
			return fmt.Errorf("routing.signals.decision[%s]: deployment %q serves %s, which answers %s questions, not %s; ask a model that declares %s questions, such as Vela 2.0",
				rule.Name, name, card.ID, answeredTypes(card), rule.Question.Type, rule.Question.Type)
		}
	}
	return nil
}

// decisionRuleDeployment returns the deployment a decision question asks and
// its definition: its own, or the decision model's shared deployment.
func (c *Classifier) decisionRuleDeployment(rule config.DecisionSignalRule) (string, config.ModelDeployment) {
	if rule.Deployment != "" {
		return rule.Deployment, c.Config.ModelDeployments[rule.Deployment]
	}
	name, deployment, _, _ := c.Config.DecisionModelDeployment()
	return name, deployment
}

func answeredTypes(card modelservice.ModelCard) string {
	if !card.Serves("decisions") {
		return "no"
	}
	types := card.QuestionTypes
	if len(types) == 0 {
		types = []string{config.DecisionQuestionChoice, config.DecisionQuestionNoul, config.DecisionQuestionScore}
	}
	return strings.Join(types, ", ")
}
