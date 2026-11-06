package config

import (
	"fmt"
)

// Remote classifier protocols describe how a built-in classifier is called.
// They are deliberately independent from the response contract.
const (
	RemoteClassifierProtocolHTTPClassify = "http_classify"
	RemoteClassifierProtocolHTTPChat     = "http_chat"
)

// Remote classifier contracts describe the semantic product returned by a
// remote classifier, independently of the protocol used to fetch it. A
// consumer declares which contracts it can read; the contract chosen tells the
// runtime how to interpret the response, so the two are not interchangeable.
// Category consumes the complete label distribution contract; PII consumes the
// token-span contract. Each built-in signal accepts exactly one shape, so the
// contract defaults per consumer and a wrong explicit value is a configuration
// error rather than a silent fallback.
const (
	RemoteClassifierContractLabelDistribution = "label_distribution.v1"
	RemoteClassifierContractLabelScores       = "label_scores.v1"
	RemoteClassifierContractLabelDecision     = "label_decision.v1"
	// RemoteClassifierContractScore carries a single continuous score for a
	// regression-style model, such as a query-difficulty scorer. It has no
	// label of its own; the consumer turns the score into a verdict.
	RemoteClassifierContractScore = "score.v1"
	// RemoteClassifierContractTokenSpans carries entity spans with code-point
	// offsets into the request string; PII is its first consumer.
	RemoteClassifierContractTokenSpans = "token_spans.v1" //nolint:gosec // G101: contract name, not a credential
)

func findNamedExternalModel(cfg *RouterConfig, name string) (*ExternalModelConfig, error) {
	if cfg == nil {
		return nil, fmt.Errorf("backend.model %q cannot be resolved without router configuration", name)
	}
	var external *ExternalModelConfig
	for i := range cfg.ExternalModels {
		if cfg.ExternalModels[i].Name != name {
			continue
		}
		if external != nil {
			return nil, fmt.Errorf("backend.model %q is ambiguous: multiple entries use this external-catalog name", name)
		}
		external = &cfg.ExternalModels[i]
	}
	if external == nil {
		return nil, fmt.Errorf("backend.model %q is not declared in global.model_catalog.external[].name", name)
	}
	return external, nil
}
