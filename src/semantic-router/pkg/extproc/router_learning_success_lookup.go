package extproc

import "strings"

type scopedExperience struct {
	scope string
	exp   routerLearningModelExperience
	found bool
}

func (s routerLearningEvidenceSnapshot) lookupChain(model string) []scopedExperience {
	model = strings.TrimSpace(model)
	if model == "" {
		return nil
	}
	hits := make([]scopedExperience, 0, 4)
	if s.cohort != "" {
		hits = append(hits, s.hit(successEvidenceScopeCohort, modelExperienceKey(s.cohort, s.tier, model)))
	}
	if s.decision != "" {
		hits = append(hits, s.hit(successEvidenceScopeDecision, modelExperienceKey(s.decision, s.tier, model)))
	}
	if s.tier != 0 {
		hits = append(hits, s.hit(successEvidenceScopeTier, modelExperienceKey("", s.tier, model)))
	}
	hits = append(hits, s.hit(successEvidenceScopeGlobal, modelExperienceKey("", 0, model)))
	return hits
}

func (s routerLearningEvidenceSnapshot) hit(scope string, key string) scopedExperience {
	exp, found := s.byKey[key]
	return scopedExperience{scope: scope, exp: exp, found: found}
}

func (s routerLearningEvidenceSnapshot) resolveExperience(model string) (routerLearningModelExperience, string, bool) {
	hits := s.lookupChain(model)
	for _, hit := range hits {
		if hit.found && !hit.exp.seedOnly() {
			return hit.exp, hit.scope, true
		}
	}
	for _, hit := range hits {
		if hit.found {
			return hit.exp, hit.scope, true
		}
	}
	return defaultRouterLearningModelExperience(), successEvidenceScopeGlobal, false
}

// mergeScopedExperience is the explicit merge path. Hierarchical lookup falls
// back instead of merging; this helper exists so conflicting populated scopes
// can be reported as `conflict` rather than a silent average.
func mergeScopedExperience(hits []scopedExperience) (routerLearningModelExperience, string, successEstimateStatus, string) {
	populated := make([]scopedExperience, 0, len(hits))
	for _, hit := range hits {
		if hit.found && !hit.exp.seedOnly() {
			populated = append(populated, hit)
		}
	}
	if len(populated) == 0 {
		return defaultRouterLearningModelExperience(), successEvidenceScopeGlobal, successEstimateInsufficient, successFallbackSeedOnly
	}
	if len(populated) == 1 {
		return populated[0].exp, populated[0].scope, "", ""
	}
	if scopedExperienceRatesConflict(populated[0].exp, populated[1].exp) {
		return routerLearningModelExperience{}, "", successEstimateConflict, successFallbackConflictingScopes
	}
	return populated[0].exp, populated[0].scope, "", ""
}
