package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type routerLearningPersonalizationScoreDiagnostic struct {
	Model        string  `json:"model"`
	Score        float64 `json:"score"`
	BaseScore    float64 `json:"base_score"`
	Preference   float64 `json:"preference"`
	Positive     float64 `json:"positive"`
	Negative     float64 `json:"negative"`
	Interactions int     `json:"interactions"`
}

type routerLearningPersonalizationDiagnostics struct {
	stateKeyHash string
	userHash     string
	selected     *routerLearningPersonalizationSelectionDetail
	preferences  []routerLearningPersonalizationScoreDiagnostic
}

type routerLearningPersonalizationSelectionDetail struct {
	model      string
	score      float64
	preference float64
}

func personalizationLearningPolicy(
	cfg config.PersonalizationLearningConfig,
	mode string,
	action routerLearningAction,
	reason string,
	stateKey string,
	userID string,
	scores []routerLearningPersonalizationScore,
	winner routerLearningPersonalizationScore,
) routerLearningPolicy {
	policy := newRouterLearningPolicy(routerLearningMethodPersonalization)
	policy.Mode = mode
	policy.Scope = cfg.EffectiveScope()
	policy.Action = action
	policy.Reason = reason
	detail := &routerLearningPersonalizationDiagnostics{}
	if stateKey != "" {
		detail.stateKeyHash = shortLearningIdentityHash(stateKey)
	}
	if userID != "" {
		detail.userHash = shortLearningIdentityHash(userID)
	}
	if winner.model != "" {
		detail.selected = &routerLearningPersonalizationSelectionDetail{
			model:      winner.model,
			score:      roundLearningFloat(winner.score),
			preference: roundLearningFloat(winner.preference),
		}
	}
	if len(scores) > 0 {
		detail.preferences = personalizationScoreDiagnostics(scores)
	}
	policy.Details.Personalization = detail
	return policy
}

func (d *routerLearningPersonalizationDiagnostics) appendLearningPolicyFields(fields *routerLearningPolicyFields) {
	if d == nil {
		return
	}
	if d.stateKeyHash != "" {
		fields.SetString(learningPolicyFieldStateKey, d.stateKeyHash)
	}
	if d.userHash != "" {
		fields.SetString(learningPolicyFieldUserHash, d.userHash)
	}
	if d.selected != nil {
		fields.SetString(learningPolicyFieldSelectedModel, d.selected.model)
		fields.SetNumber(learningPolicyFieldSelectedScore, d.selected.score)
		fields.SetNumber(learningPolicyFieldSelectedPref, d.selected.preference)
	}
	if len(d.preferences) > 0 {
		fields.Set(learningPolicyFieldPreferences, append([]routerLearningPersonalizationScoreDiagnostic(nil), d.preferences...))
	}
}

func personalizationScoreDiagnostics(scores []routerLearningPersonalizationScore) []routerLearningPersonalizationScoreDiagnostic {
	result := make([]routerLearningPersonalizationScoreDiagnostic, 0, len(scores))
	for _, score := range scores {
		result = append(result, routerLearningPersonalizationScoreDiagnostic{
			Model:        score.model,
			Score:        roundLearningFloat(score.score),
			BaseScore:    roundLearningFloat(score.baseScore),
			Preference:   roundLearningFloat(score.preference),
			Positive:     roundLearningFloat(score.positive),
			Negative:     roundLearningFloat(score.negative),
			Interactions: score.interactions,
		})
	}
	return result
}
