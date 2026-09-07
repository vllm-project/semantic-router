package config

import "fmt"

// ComplexityBackendContracts are the response shapes the complexity signal can
// read. Both are supported because a difficulty model can be built either way,
// and the contract chosen decides how the runtime reads the response:
//
//   - score.v1 carries a continuous score, which the signal turns into a
//     verdict using each rule's declared boundaries.
//   - label_distribution.v1 carries the verdict directly as a label, so the
//     winning label is the verdict and its probability is the confidence.
//
// Neither is a substitute for the other, which is why the field cannot be
// defaulted for this signal.
var ComplexityBackendContracts = []string{
	RemoteClassifierContractScore,
	RemoteClassifierContractLabelDistribution,
}

// ValidateComplexityModelBackend checks the complexity signal's remote
// attachment. A nil backend is the local prototype-scoring path and stays
// valid, so this reports an error only for a configured backend the runtime
// could not honour.
func ValidateComplexityModelBackend(cfg *RouterConfig) error {
	if cfg == nil {
		return fmt.Errorf("complexity model configuration is nil")
	}
	if err := ValidateComplexityRuleBoundaries(cfg); err != nil {
		return err
	}
	backend := cfg.ComplexityModel.Backend
	if backend == nil {
		return nil
	}
	// http_chat returns prose. Reading either a score or a label distribution
	// out of it needs a parser this signal does not define, so it is rejected
	// here rather than failing per request.
	if backend.Protocol != RemoteClassifierProtocolHTTPClassify {
		return fmt.Errorf(
			"complexity.backend.protocol %q is not supported by the complexity consumer, use %q",
			backend.Protocol, RemoteClassifierProtocolHTTPClassify)
	}
	if _, err := ResolveRemoteClassifierBackend(
		cfg,
		backend,
		ModelRoleClassification,
		ComplexityBackendContracts...,
	); err != nil {
		return fmt.Errorf("complexity: %w", err)
	}
	return nil
}

// ValidateComplexityRuleBoundaries checks every complexity rule's declared
// boundaries, including the rules that exist only inside a recipe, since
// routing.signals is replaced wholesale per recipe.
//
// A rule may only declare the lower-is-harder pair when a score backend
// supplies the number. The local margin is hardScore minus easyScore, so a
// higher value is harder by construction: honouring the opposite direction
// there would invert every verdict the rule reaches while the config still
// looked reasonable. Locally, the inverse is expressed by swapping the
// candidate lists.
func ValidateComplexityRuleBoundaries(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	scored := complexityScoreBackendConfigured(cfg)
	for _, ref := range complexityRuleRefs(cfg) {
		bounds, err := ref.rule.EffectiveBoundaries()
		if err != nil {
			return fmt.Errorf("%s%w", ref.prefix(), err)
		}
		if !bounds.HigherIsHarder && !scored {
			return fmt.Errorf(
				"%scomplexity rule %q declares hard_below/easy_above, which needs a %s backend: "+
					"the local margin is hard-minus-easy, so a higher score is harder by construction. "+
					"To invert it locally, swap the hard and easy candidate lists",
				ref.prefix(), ref.rule.Name, RemoteClassifierContractScore)
		}
	}
	return nil
}

func complexityScoreBackendConfigured(cfg *RouterConfig) bool {
	backend := cfg.ComplexityModel.Backend
	return backend != nil && backend.EffectiveContract("") == RemoteClassifierContractScore
}

type complexityRuleRef struct {
	recipe RecipeName
	rule   ComplexityRule
}

func (r complexityRuleRef) prefix() string {
	if r.recipe == "" {
		return ""
	}
	return fmt.Sprintf("recipe %q: ", r.recipe)
}

func complexityRuleRefs(cfg *RouterConfig) []complexityRuleRef {
	refs := make([]complexityRuleRef, 0, len(cfg.ComplexityRules))
	for _, rule := range cfg.ComplexityRules {
		refs = append(refs, complexityRuleRef{rule: rule})
	}
	for _, recipe := range cfg.Recipes {
		for _, rule := range recipe.Profile.Signals.ComplexityRules {
			refs = append(refs, complexityRuleRef{recipe: recipe.Name, rule: rule})
		}
	}
	return refs
}
