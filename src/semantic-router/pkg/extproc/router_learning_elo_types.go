package extproc

import "time"

const (
	routerLearningMethodElo routerLearningMethod = "elo"

	eloReasonBaseBest        = "base_best"
	eloReasonDecisionBypass  = "decision_bypass"
	eloReasonIdentityMissing = "identity_missing"
	eloReasonNoCandidates    = "no_candidates"
	eloReasonRatingWin       = "rating_win"
	eloReasonStateMissing    = "state_missing"
)

type routerLearningEloState struct {
	ratings map[string]map[string]*routerLearningEloRating
}

type routerLearningEloRating struct {
	Model       string
	Rating      float64
	Comparisons int
	Wins        int
	Losses      int
	Ties        int
	LastUpdated time.Time
}

type routerLearningEloScore struct {
	model       string
	rating      float64
	score       float64
	comparisons int
	wins        int
	losses      int
	ties        int
	known       bool
}
