package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type routerLearningBanditScoreDiagnostic struct {
	Model         string  `json:"model"`
	Score         float64 `json:"score"`
	Quality       float64 `json:"quality"`
	Cost          float64 `json:"cost"`
	Latency       float64 `json:"latency"`
	RewardMean    float64 `json:"reward_mean"`
	Exploration   float64 `json:"exploration"`
	Impressions   int     `json:"impressions"`
	FeedbackCount int     `json:"feedback_count"`
}

type routerLearningBanditDetail struct {
	algorithm    string
	goals        map[string]float64
	stateKeyHash string
	selected     *routerLearningBanditSelectionDetail
	scores       []routerLearningBanditScoreDiagnostic
}

type routerLearningBanditSelectionDetail struct {
	model string
	score float64
}

func banditLearningPolicy(
	cfg config.BanditLearningConfig,
	mode string,
	action routerLearningAction,
	reason string,
	stateKey string,
	scores []routerLearningBanditScore,
	winner routerLearningBanditScore,
) routerLearningPolicy {
	policy := newRouterLearningPolicy(routerLearningMethodBandit)
	policy.Mode = mode
	policy.Scope = cfg.EffectiveScope()
	policy.Action = action
	policy.Reason = reason
	detail := &routerLearningBanditDetail{
		algorithm: cfg.EffectiveAlgorithm(),
		goals:     normalizedLearningGoals(cfg.Goals),
	}
	if stateKey != "" {
		detail.stateKeyHash = shortLearningIdentityHash(stateKey)
	}
	if winner.model != "" {
		detail.selected = &routerLearningBanditSelectionDetail{
			model: winner.model,
			score: roundLearningFloat(winner.score),
		}
	}
	if len(scores) > 0 {
		detail.scores = banditScoreDiagnostics(scores)
	}
	policy.Detail = detail
	return policy
}

func (d *routerLearningBanditDetail) appendLearningPolicyFields(out map[string]interface{}) {
	if d == nil {
		return
	}
	if d.algorithm != "" {
		out["algorithm"] = d.algorithm
	}
	if len(d.goals) > 0 {
		out["goals"] = cloneLearningGoals(d.goals)
	}
	if d.stateKeyHash != "" {
		out["state_key_hash"] = d.stateKeyHash
	}
	if d.selected != nil {
		out["selected_model"] = d.selected.model
		out["selected_score"] = d.selected.score
	}
	if len(d.scores) > 0 {
		out["scores"] = append([]routerLearningBanditScoreDiagnostic(nil), d.scores...)
	}
}

func banditScoreDiagnostics(scores []routerLearningBanditScore) []routerLearningBanditScoreDiagnostic {
	result := make([]routerLearningBanditScoreDiagnostic, 0, len(scores))
	for _, score := range scores {
		result = append(result, routerLearningBanditScoreDiagnostic{
			Model:         score.model,
			Score:         roundLearningFloat(score.score),
			Quality:       roundLearningFloat(score.quality),
			Cost:          roundLearningFloat(score.cost),
			Latency:       roundLearningFloat(score.latency),
			RewardMean:    roundLearningFloat(score.meanReward),
			Exploration:   roundLearningFloat(score.exploration),
			Impressions:   score.impressions,
			FeedbackCount: score.feedbackCount,
		})
	}
	return result
}
