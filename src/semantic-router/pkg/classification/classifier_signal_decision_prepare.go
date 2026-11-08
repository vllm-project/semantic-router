package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// labelledDecisionRules returns used custom questions for structural preparation.
func (c *Classifier) labelledDecisionRules() []config.DecisionSignalRule {
	if c == nil || c.Config == nil || !c.usesRoutingSignalType(config.SignalTypeDecision) {
		return nil
	}
	used := c.getUsedSignals()
	var rules []config.DecisionSignalRule
	for _, rule := range c.Config.DecisionRules {
		if signalRuleUsed(used, config.SignalTypeDecision, rule.Name) {
			rules = append(rules, rule)
		}
	}
	return rules
}

// prepareDecisionSignals checks native or composed implementations before
// publishing the generation. Without runtime services there is no card of a
// model_runtime deployment to check. A systemone deployment's questions are
// checked against the question types it declares; its API's listing may add
// more, so a deployment that declares none is checked per request.
func (c *Classifier) prepareDecisionSignals() error {
	services := c.models != nil && c.models.runtime != nil && c.models.runtime.Services() != nil
	cards := make(map[string]modelservice.ModelCard)
	for _, rule := range c.labelledDecisionRules() {
		name, deployment := c.decisionRuleDeployment(rule)
		if deployment.IsSystemOne() {
			declared := deployment.DeclaredQuestionTypes()
			if len(declared) == 0 {
				continue
			}
			if _, err := modelservice.CompileTask(modelservice.TaskDefinition{ID: "decision", Stage: "request"}, DecisionQuestion(rule.Name, rule.Question), modelservice.RemoteCard(deployment.Model, declared, deployment.DeclaredPresets())); err != nil {
				return fmt.Errorf("routing.signals.decision[%s]: systemone deployment %q: %w", rule.Name, name, err)
			}
			continue
		}
		if !services {
			continue
		}
		card, checked := cards[name]
		if !checked {
			if deployment.Managed() {
				var err error
				card, err = c.models.runtime.DeploymentCard(context.Background(), name, deployment)
				if err != nil {
					return fmt.Errorf("routing.signals.decision[%s]: %w", rule.Name, err)
				}
			} else if observed, ready := c.models.runtime.CurrentDeploymentCard(name); ready {
				card = observed
			} else {
				// Its watcher owns discovery. Requests remain unknown until a
				// ready card can compile this same question.
				continue
			}
			cards[name] = card
		}
		if _, err := modelservice.CompileTask(modelservice.TaskDefinition{ID: "decision", Stage: "request"}, DecisionQuestion(rule.Name, rule.Question), card); err != nil {
			return fmt.Errorf("routing.signals.decision[%s]: deployment %q: %w", rule.Name, name, err)
		}
	}
	if services {
		c.decisionCards = cards
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

// decisionTaskCard reads prepared managed metadata or this generation's current
// attached observation. Request-time execution never discovers a model.
func (c *Classifier) decisionTaskCard(deployment string) (modelservice.ModelCard, bool) {
	if c.models != nil && c.models.runtime != nil && !c.Config.ModelDeployments[deployment].Managed() {
		return c.models.runtime.CurrentDeploymentCard(deployment)
	}
	card, ok := c.decisionCards[deployment]
	return card, ok
}
