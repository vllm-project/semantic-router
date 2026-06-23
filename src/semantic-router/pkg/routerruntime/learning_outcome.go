package routerruntime

import "context"

type (
	RouterOutcomeSource  string
	RouterOutcomeTarget  string
	RouterOutcomeVerdict string
)

const (
	RouterOutcomeSourceUser     RouterOutcomeSource = "user"
	RouterOutcomeSourceAgent    RouterOutcomeSource = "agent"
	RouterOutcomeSourceEval     RouterOutcomeSource = "eval"
	RouterOutcomeSourceOperator RouterOutcomeSource = "operator"
	RouterOutcomeSourceProvider RouterOutcomeSource = "provider"
	RouterOutcomeSourceRouter   RouterOutcomeSource = "router"

	RouterOutcomeTargetModel     RouterOutcomeTarget = "model"
	RouterOutcomeTargetRoute     RouterOutcomeTarget = "route"
	RouterOutcomeTargetPolicy    RouterOutcomeTarget = "policy"
	RouterOutcomeTargetStability RouterOutcomeTarget = "stability"
	RouterOutcomeTargetProvider  RouterOutcomeTarget = "provider"
	RouterOutcomeTargetRouter    RouterOutcomeTarget = "router"

	RouterOutcomeVerdictGoodFit         RouterOutcomeVerdict = "good_fit"
	RouterOutcomeVerdictUnderpowered    RouterOutcomeVerdict = "underpowered"
	RouterOutcomeVerdictOverprovisioned RouterOutcomeVerdict = "overprovisioned"
	RouterOutcomeVerdictFailed          RouterOutcomeVerdict = "failed"
)

type RouterOutcome struct {
	ReplayID  string
	Source    RouterOutcomeSource
	Target    RouterOutcomeTarget
	TargetRef string
	Verdict   RouterOutcomeVerdict
	Reason    string
	Score     float64
	Metadata  map[string]string
}

type RouterOutcomeResult struct {
	Updated  int
	Recorded bool
}

type OutcomeRuntime interface {
	UpdateOutcome(context.Context, *RouterOutcome) RouterOutcomeResult
}
