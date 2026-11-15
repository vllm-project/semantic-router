package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// preparedSystemOneRules are the systemone and application rules a
// generation prepares: the request-stage rules its decisions use, every
// response-stage rule, which a response plugin asks, and the hallucination
// question a grounding fusion asks about each answer.
func (c *Classifier) preparedSystemOneRules() []askedRule {
	if c == nil || c.Config == nil {
		return nil
	}
	used := c.getUsedSignals()
	var rules []askedRule
	for _, rule := range c.Config.SystemOneRules {
		if rule.EffectiveStage() == config.SystemOneStageResponse || signalRuleUsed(used, config.SignalTypeSystemOne, rule.Name) {
			rules = append(rules, systemOneAsked(rule))
		}
	}
	for _, rule := range c.applicationRules {
		if rule.Rule.EffectiveStage() == config.SystemOneStageResponse || signalRuleUsed(used, rule.Type, rule.Rule.Name) {
			rules = append(rules, applicationAsked(rule))
		}
	}
	if decision, grounds := c.Config.GroundingFusionDecision(); grounds {
		if asked, _, err := c.presetAsked(config.SignalTypeHallucination, presets.Hallucination, presets.StageResponse); err == nil {
			asked.path = c.Config.ScopePath("decisions[" + decision + "].algorithm.fusion.grounding")
			rules = append(rules, asked)
		}
	}
	return rules
}

// prepareSystemOneSignals checks native or composed implementations before
// publishing the generation, and what the load could not tell about the
// deployment that answers each rule (config.CheckSystemOneAnswerer). Without
// runtime services there is no card of a model_runtime deployment to check.
// A systemone deployment's questions are checked against the card its API
// lists, read now, else the question types, presets and modalities it
// declares, and a card without modalities reads text; the question types of
// one whose API lists none and that declares none are checked per request.
func (c *Classifier) prepareSystemOneSignals() error {
	services := c.models != nil && c.models.runtime != nil && c.models.runtime.Services() != nil
	cards := make(map[string]modelservice.ModelCard)
	for _, asked := range c.preparedSystemOneRules() {
		rule := asked.rule
		name, deployment := c.systemOneRuleDeployment(rule)
		path := c.Config.ScopePath("signals.systemone[" + rule.Name + "]")
		definition := modelservice.TaskDefinition{ID: config.SignalTypeSystemOne, Stage: rule.EffectiveStage()}
		check := func(card modelservice.ModelCard, questionTypes []string) error {
			return config.CheckSystemOneAnswerer(path, name, rule, questionTypes, card.ReadModalities())
		}
		if asked.preset != "" {
			path = c.Config.ApplicationRulePath(asked.signalType, rule.Name)
			definition = modelservice.TaskDefinition{ID: modelservice.PresetTask(asked.preset), Stage: rule.EffectiveStage(), Preset: asked.preset}
			application := config.ApplicationRule{Type: asked.signalType, Preset: asked.preset, Rule: rule}
			check = func(card modelservice.ModelCard, questionTypes []string) error {
				return config.CheckApplicationAnswerer(path, c.Config.ScopePath(""), name, application, questionTypes, card.Presets, card.ReadModalities())
			}
		}
		if asked.path != "" {
			path = asked.path
		}
		question := askedQuestion(asked)
		if deployment.IsSystemOne() {
			card := modelservice.DeploymentCard(deployment)
			if remote := c.systemOne(); remote != nil {
				var err error
				if card, err = remote.PrepareCard(context.Background(), name); err != nil {
					return fmt.Errorf("%s: %w", path, err)
				}
			}
			var questionTypes []string
			if len(card.QuestionTypes) > 0 {
				questionTypes = card.AnsweredTypes()
			}
			if err := check(card, questionTypes); err != nil {
				return err
			}
			if questionTypes == nil {
				continue
			}
			if _, err := modelservice.CompileTask(definition, question, card); err != nil {
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
		if err := check(card, card.AnsweredTypes()); err != nil {
			return err
		}
		if _, err := modelservice.CompileTask(definition, question, card); err != nil {
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
