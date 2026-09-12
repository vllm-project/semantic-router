package config

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

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
	if err := validateComplexityRoutingContracts(cfg); err != nil {
		return err
	}
	return validateComplexityModelBackendContracts(cfg)
}

// validateComplexityModelBackendContracts checks the static attachment before
// startup. Rule boundaries are checked separately once routing state is ready.
func validateComplexityModelBackendContracts(cfg *RouterConfig) error {
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
		// A remote score arrives in the model's own units. `threshold` is
		// symmetric about zero, which only means something for the local
		// signed margin - against a [0,1] scorer it puts every request past
		// the hard cut and makes easy unreachable. Declaring nothing collapses
		// both cut points onto zero, with the same effect. Either way the
		// config looks reasonable and the verdicts are wrong, so require the
		// explicit pair.
		if scored && !ref.rule.declaresBoundaryPair() {
			return fmt.Errorf(
				"%scomplexity rule %q needs an explicit boundary pair under %s: "+
					"a remote score is in the model's own units, so threshold - which is symmetric "+
					"about zero for the local margin - cannot convert it. Use hard_above with "+
					"easy_below, or hard_below with easy_above",
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

// ComplexityBackendAdvisories reports consequences of a complexity backend
// that are correct but easy to miss. They are advisories rather than errors:
// each describes a configuration the runtime will honour exactly as written,
// where the surprise is what the config no longer does.
//
// It returns the messages instead of logging them so the reasoning is
// testable without capturing log output; ValidateComplexityModelBackend emits
// them.
func ComplexityBackendAdvisories(cfg *RouterConfig) []string {
	if cfg == nil || cfg.ComplexityModel.Backend == nil {
		return nil
	}
	advisories := make([]string, 0, 2)

	// score.v1 reports no confidence by design, so any decision gated on one
	// of these rules drops out of confidence-based ranking and competes on
	// the engine's structural default instead. A rule alongside it on the
	// local path still reports one, so the change is invisible in the config
	// but visible in which decision wins.
	if cfg.ComplexityModel.Backend.Contract == RemoteClassifierContractScore {
		advisories = append(advisories, fmt.Sprintf(
			"complexity.backend uses %s, which reports no confidence: decisions gated on a complexity rule "+
				"will rank on the engine's structural default rather than a reported score. "+
				"%s reports the winning label's probability if confidence-based ranking matters",
			RemoteClassifierContractScore, RemoteClassifierContractLabelDistribution))
	}

	// The candidate lists are how the local path produces a score. With a
	// backend they are never read, so editing them has no effect - worth
	// saying once rather than leaving someone to discover it.
	var withCandidates []string
	for _, ref := range complexityRuleRefs(cfg) {
		if len(ref.rule.Hard.Candidates) > 0 || len(ref.rule.Easy.Candidates) > 0 ||
			len(ref.rule.Hard.ImageCandidates) > 0 || len(ref.rule.Easy.ImageCandidates) > 0 {
			withCandidates = append(withCandidates, ref.prefix()+ref.rule.Name)
		}
	}
	if len(withCandidates) > 0 {
		advisories = append(advisories, fmt.Sprintf(
			"complexity.backend supplies the score, so the hard/easy candidate lists on %s are never read; "+
				"they can be removed",
			strings.Join(withCandidates, ", ")))
	}
	return advisories
}

// validateComplexityRoutingContracts checks all recipe rules against the shared
// backend and reports advisories once the complete routing graph is available.
func validateComplexityRoutingContracts(cfg *RouterConfig) error {
	if err := ValidateComplexityRuleBoundaries(cfg); err != nil {
		return err
	}
	for _, advisory := range ComplexityBackendAdvisories(cfg) {
		logging.Warnf("%s", advisory)
	}
	return nil
}
