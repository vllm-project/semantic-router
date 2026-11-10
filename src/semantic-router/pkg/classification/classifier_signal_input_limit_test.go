package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// NewSignalErrorTestClassifier is available only to external tests: its
// jailbreak and pii rules ask a decision model whose every call fails with
// err, and each rule's decision fails the request when the rule is unknown.
func NewSignalErrorTestClassifier(err error) (*Classifier, error) {
	cfg := &config.RouterConfig{}
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "guard", Deployment: "guard-model"}}
	cfg.PIIRules = []config.PIIRule{{Name: "private", Deployment: "guard-model"}}
	cfg.Decisions = []config.Decision{
		{Name: "guarded", Priority: 2, Rules: config.RuleNode{Type: config.SignalTypeJailbreak, Name: "guard", OnUnknown: config.RuleOnUnknownFailRequest}},
		{Name: "private", Priority: 1, Rules: config.RuleNode{Type: config.SignalTypePII, Name: "private", OnUnknown: config.RuleOnUnknownFailRequest}},
	}
	classifier, buildErr := newClassifierWithOptions(cfg)
	if buildErr != nil {
		return nil, buildErr
	}
	decider := newFakeDecider(nil)
	decider.failures["guard-model"] = err
	classifier.SetDecisionDecider(decider)
	return classifier, nil
}
