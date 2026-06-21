package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type routerLearningEloScoreDiagnostic struct {
	Model       string  `json:"model"`
	Score       float64 `json:"score"`
	Rating      float64 `json:"rating"`
	Comparisons int     `json:"comparisons"`
	Wins        int     `json:"wins"`
	Losses      int     `json:"losses"`
	Ties        int     `json:"ties"`
}

type routerLearningEloDiagnostics struct {
	initialRating float64
	kFactor       float64
	stateKeyHash  string
	selected      *routerLearningEloSelectionDetail
	ratings       []routerLearningEloScoreDiagnostic
}

type routerLearningEloSelectionDetail struct {
	model  string
	score  float64
	rating float64
}

func eloLearningPolicy(
	cfg config.EloLearningConfig,
	mode string,
	action routerLearningAction,
	reason string,
	stateKey string,
	scores []routerLearningEloScore,
	winner routerLearningEloScore,
) routerLearningPolicy {
	policy := newRouterLearningPolicy(routerLearningMethodElo)
	policy.Mode = mode
	policy.Scope = cfg.EffectiveScope()
	policy.Action = action
	policy.Reason = reason
	detail := &routerLearningEloDiagnostics{
		initialRating: roundLearningFloat(eloInitialRating(cfg)),
		kFactor:       roundLearningFloat(eloKFactor(cfg)),
	}
	if stateKey != "" {
		detail.stateKeyHash = shortLearningIdentityHash(stateKey)
	}
	if winner.model != "" {
		detail.selected = &routerLearningEloSelectionDetail{
			model:  winner.model,
			score:  roundLearningFloat(winner.score),
			rating: roundLearningFloat(winner.rating),
		}
	}
	if len(scores) > 0 {
		detail.ratings = eloScoreDiagnostics(scores)
	}
	policy.Details.Elo = detail
	return policy
}

func (d *routerLearningEloDiagnostics) appendLearningPolicyFields(fields *routerLearningPolicyFields) {
	if d == nil {
		return
	}
	fields.SetNumber(learningPolicyFieldInitialRating, d.initialRating)
	fields.SetNumber(learningPolicyFieldKFactor, d.kFactor)
	if d.stateKeyHash != "" {
		fields.SetString(learningPolicyFieldStateKey, d.stateKeyHash)
	}
	if d.selected != nil {
		fields.SetString(learningPolicyFieldSelectedModel, d.selected.model)
		fields.SetNumber(learningPolicyFieldSelectedScore, d.selected.score)
		fields.SetNumber(learningPolicyFieldSelectedRating, d.selected.rating)
	}
	if len(d.ratings) > 0 {
		fields.Set(learningPolicyFieldRatings, append([]routerLearningEloScoreDiagnostic(nil), d.ratings...))
	}
}

func eloScoreDiagnostics(scores []routerLearningEloScore) []routerLearningEloScoreDiagnostic {
	result := make([]routerLearningEloScoreDiagnostic, 0, len(scores))
	for _, score := range scores {
		result = append(result, routerLearningEloScoreDiagnostic{
			Model:       score.model,
			Score:       roundLearningFloat(score.score),
			Rating:      roundLearningFloat(score.rating),
			Comparisons: score.comparisons,
			Wins:        score.wins,
			Losses:      score.losses,
			Ties:        score.ties,
		})
	}
	return result
}
