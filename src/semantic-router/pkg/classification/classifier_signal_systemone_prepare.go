package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// preparedSystemOneRules are the rules a generation prepares: the
// request-stage rules its decisions use, and every response-stage rule, which
// a response plugin asks.
func (c *Classifier) preparedSystemOneRules() []config.SystemOneRule {
	if c == nil || c.Config == nil {
		return nil
	}
	var used map[string]bool
	if c.usesRoutingSignalType(config.SignalTypeSystemOne) {
		used = c.getUsedSignals()
	}
	var rules []config.SystemOneRule
	for _, rule := range c.Config.SystemOneRules {
		if rule.EffectiveStage() == config.SystemOneStageResponse || signalRuleUsed(used, config.SignalTypeSystemOne, rule.Name) {
			rules = append(rules, rule)
		}
	}
	return rules
}

// prepareSystemOneSignals checks native or composed implementations before
// publishing the generation, and what the load could not tell about the
// deployment that answers each rule (config.CheckSystemOneAnswerer). Without
// runtime services there is no card of a model_runtime deployment to check.
// A systemone deployment's questions are checked against the question types
// and modalities it declares; its API's listing may add more, so a
// deployment that declares none is checked per request.
func (c *Classifier) prepareSystemOneSignals() error {
	services := c.models != nil && c.models.runtime != nil && c.models.runtime.Services() != nil
	cards := make(map[string]modelservice.ModelCard)
	for _, rule := range c.preparedSystemOneRules() {
		name, deployment := c.systemOneRuleDeployment(rule)
		path := c.Config.ScopePath("signals.systemone[" + rule.Name + "]")
		definition := modelservice.TaskDefinition{ID: config.SignalTypeSystemOne, Stage: rule.EffectiveStage()}
		if deployment.IsSystemOne() {
			declared := deployment.DeclaredQuestionTypes()
			if len(declared) == 0 {
				continue
			}
			var modalities []string
			if len(deployment.DeclaredModalities()) > 0 {
				modalities = deployment.DeclaredModalities()
			}
			if err := config.CheckSystemOneAnswerer(path, name, rule, declared, modalities); err != nil {
				return err
			}
			if _, err := modelservice.CompileTask(definition, systemOneRuleQuestion(rule), modelservice.RemoteCard(deployment.Model, declared, deployment.DeclaredPresets())); err != nil {
				return fmt.Errorf("%s: systemone deployment %q: %w", path, name, err)
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
					return fmt.Errorf("%s: %w", path, err)
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
		if err := config.CheckSystemOneAnswerer(path, name, rule, card.AnsweredTypes(), card.ReadModalities()); err != nil {
			return err
		}
		if _, err := modelservice.CompileTask(definition, systemOneRuleQuestion(rule), card); err != nil {
			return fmt.Errorf("%s: deployment %q: %w", path, name, err)
		}
	}
	if services {
		c.decisionCards = cards
	}
	return nil
}

// systemOneRuleDeployment returns the deployment a rule asks and its
// definition: its own, or the decision model's shared deployment.
func (c *Classifier) systemOneRuleDeployment(rule config.SystemOneRule) (string, config.ModelDeployment) {
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
