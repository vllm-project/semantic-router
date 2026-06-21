package extproc

import "time"

const (
	routerLearningMethodPersonalization routerLearningMethod = "personalization"

	personalizationReasonBaseBest        = "base_best"
	personalizationReasonDecisionBypass  = "decision_bypass"
	personalizationReasonIdentityMissing = "identity_missing"
	personalizationReasonNoCandidates    = "no_candidates"
	personalizationReasonPreferenceWin   = "preference_win"
	personalizationReasonStateMissing    = "state_missing"
)

type routerLearningPersonalizationState struct {
	preferences map[string]map[string]*routerLearningPersonalizationModelState
}

type routerLearningPersonalizationModelState struct {
	Model        string
	Positive     float64
	Negative     float64
	Interactions int
	LastUpdated  time.Time
}

type routerLearningPersonalizationScore struct {
	model        string
	baseScore    float64
	preference   float64
	score        float64
	positive     float64
	negative     float64
	interactions int
	known        bool
}
