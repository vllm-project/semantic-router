package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// A request stage sends every question it asks one decision model in one
// call (modelservice.Bundle): each model-backed signal joins the stage's
// bundle naming the deployments it asks, as its prepared bindings and the
// used decision rules say, so a deployment's call waits for exactly the
// signals that ask it and never for the others.

// questionAsker is a prepared backend that asks a decision model its
// signal's question; it names the deployment, or "" when it asks none.
type questionAsker interface {
	questionDeployment() string
}

// askedDeployment is the deployment a prepared binding asks a Preset or
// Question of, or "".
func askedDeployment(capability binding.Capability) string {
	if capability.Question == "" && capability.Preset == "" {
		return ""
	}
	return capability.Deployment
}

func (b *ownedSequenceBackend) questionDeployment() string {
	if b == nil {
		return ""
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.handle == nil {
		return ""
	}
	return askedDeployment(b.handle.Capability())
}

func (b *ownedTokenBackend) questionDeployment() string {
	if b == nil {
		return ""
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.handle == nil {
		return ""
	}
	return askedDeployment(b.handle.Capability())
}

func (m *ownedModalityClassifier) questionDeployment() string {
	if m == nil || m.handle == nil {
		return ""
	}
	return askedDeployment(m.handle.Capability())
}

func (c *ownedLabelTask[I, O]) questionDeployment() string {
	if c == nil {
		return ""
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.handle == nil {
		return ""
	}
	return askedDeployment(c.handle.Capability())
}

// signalQuestionDeployments lists the deployments a signal asks questions of
// in a request stage: its prepared bindings' and, for decision signals, those
// of the used rules.
func (c *Classifier) signalQuestionDeployments(signalType string, usedSignals map[string]bool) []string {
	if c == nil {
		return nil
	}
	var consumers []interface{}
	switch signalType {
	case config.SignalTypeDecision:
		var deployments []string
		for _, rule := range c.Config.DecisionRules {
			if signalRuleUsed(usedSignals, config.SignalTypeDecision, rule.Name) {
				if deployment := c.Config.DecisionQuestionDeployment(rule); deployment != "" {
					deployments = append(deployments, deployment)
				}
			}
		}
		return deployments
	case config.SignalTypeDomain:
		consumers = append(consumers, c.categoryInference)
	case config.SignalTypePreference:
		if c.preferenceClassifier != nil && c.preferenceClassifier.judgment != nil {
			consumers = append(consumers, c.preferenceClassifier.judgment)
		}
	case config.SignalTypeComplexity:
		if c.complexityClassifier != nil {
			for _, judgment := range c.complexityClassifier.judgments {
				consumers = append(consumers, judgment)
			}
		}
	case config.SignalTypeClassifier:
		for name, classifier := range c.genericClassifiers {
			if signalRuleUsed(usedSignals, signalType, name) {
				consumers = append(consumers, classifier)
			}
		}
	case config.SignalTypeJailbreak:
		consumers = append(consumers, c.jailbreakInference)
	case config.SignalTypeFactCheck:
		if c.factCheckClassifier != nil {
			consumers = append(consumers, c.factCheckClassifier.backend)
		}
	case config.SignalTypeUserFeedback:
		if c.feedbackDetector != nil {
			consumers = append(consumers, c.feedbackDetector.backend)
		}
	case config.SignalTypeModality:
		consumers = append(consumers, c.modalityInference)
	case config.SignalTypeReask:
		if c.reaskClassifier != nil && c.reaskClassifier.judgment != nil {
			return []string{c.reaskClassifier.judgment.deployment}
		}
	case config.SignalTypePII:
		consumers = append(consumers, c.piiInference)
	case config.SignalTypeSafety:
		for _, detector := range c.safetyClassifiers {
			if detector != nil {
				consumers = append(consumers, detector.binary)
				consumers = append(consumers, detector.hazard)
			}
		}
	}
	var deployments []string
	for _, consumer := range consumers {
		if asker, ok := consumer.(questionAsker); ok {
			if deployment := asker.questionDeployment(); deployment != "" {
				deployments = append(deployments, deployment)
			}
		}
	}
	return deployments
}
