package classification

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// A request stage sends every question it asks one decision model in one
// call (modelservice.Bundle): each systemone and application signal joins
// the stage's bundle naming the deployments its used request-stage rules
// ask, so a deployment's call waits for exactly the signals that ask it and
// never for the others.

// signalQuestionDeployments lists the deployments a signal type's used
// request-stage rules ask.
func (c *Classifier) signalQuestionDeployments(signalType string, usedSignals map[string]bool) []string {
	if c == nil {
		return nil
	}
	var rules []config.SystemOneRule
	if signalType == config.SignalTypeSystemOne {
		rules = c.Config.SystemOneRules
	}
	for _, rule := range c.applicationRules {
		if rule.Type == signalType {
			rules = append(rules, rule.Rule)
		}
	}
	var deployments []string
	for _, rule := range rules {
		if rule.EffectiveStage() == config.SystemOneStageRequest && signalRuleUsed(usedSignals, signalType, rule.Name) {
			if deployment := c.Config.SystemOneRuleDeployment(rule); deployment != "" {
				deployments = append(deployments, deployment)
			}
		}
	}
	return deployments
}
