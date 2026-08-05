package extproc

type routerLearningOutcomeVerdict string

const (
	routerLearningOutcomeGoodFit         routerLearningOutcomeVerdict = "good_fit"
	routerLearningOutcomeUnderpowered    routerLearningOutcomeVerdict = "underpowered"
	routerLearningOutcomeOverprovisioned routerLearningOutcomeVerdict = "overprovisioned"
	routerLearningOutcomeFailed          routerLearningOutcomeVerdict = "failed"
)
