package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// newContextualDiagnostics builds a routerLearningAdaptationDiagnostics for
// contextual-bandit strategies (LinUCB / Linear Thompson). It reuses the
// existing diagnostics struct so replay/header serialization paths do not
// need a new method-keyed branch — the strategy field on the diagnostic is
// the only thing that distinguishes routing_sampling output from a
// contextual run, and per-arm score components keep the same names so
// downstream tools work uniformly.
//
// `dim`, `alpha`, `lambda`, `sigma` are stashed in the score-level
// diagnostics map so the replay record is self-describing — a reader can
// see "this row used dim=64 alpha=1.0 lambda=1.0" without consulting the
// router config at the corresponding wall-clock time.
func newContextualDiagnostics(
	learningCtx *selection.SelectionContext,
	ctx *RequestContext,
	candidateSet string,
	strategy string,
	baseModel string,
	winner routerLearningCandidateScore,
	usedSampling bool,
	scores []routerLearningCandidateScore,
	dim int,
	alpha float64,
	lambda float64,
	sigma float64,
) *routerLearningAdaptationDiagnostics {
	diag := &routerLearningAdaptationDiagnostics{
		candidateSet:  candidateSet,
		strategy:      strategy,
		baseModel:     baseModel,
		proposalModel: winner.model,
		decision:      strings.TrimSpace(learningCtx.DecisionName),
		decisionTier:  decisionTier(ctx),
		sampling: routerLearningSamplingDiagnostics{
			used: usedSampling,
		},
		scores: scores,
	}
	diag.dim = dim
	diag.alpha = alpha
	diag.lambda = lambda
	diag.sigma = sigma
	return diag
}
