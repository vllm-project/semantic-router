package extproc

import "time"

const (
	routerLearningMethodBandit routerLearningMethod = "bandit"

	banditReasonBaseBest        = "base_best"
	banditReasonDecisionBypass  = "decision_bypass"
	banditReasonIdentityMissing = "identity_missing"
	banditReasonNoCandidates    = "no_candidates"
	banditReasonScoreWin        = "score_win"
	banditReasonStateMissing    = "state_missing"
)

type routerLearningBanditState struct {
	arms map[string]map[string]*routerLearningBanditArmState
}

type routerLearningBanditArmState struct {
	Impressions   int
	FeedbackCount int
	RewardSum     float64
	LastUpdated   time.Time
}

type routerLearningBanditScore struct {
	model            string
	quality          float64
	cost             float64
	latency          float64
	meanReward       float64
	exploration      float64
	score            float64
	impressions      int
	feedbackCount    int
	knownRewardState bool
}
