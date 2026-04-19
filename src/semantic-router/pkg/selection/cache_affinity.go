package selection

import (
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// CacheAffinityContext carries the request-time session signals used by the
// pre-dispatch cache-affinity estimator.
//
// These fields capture runtime evidence for the current request. The selection
// request path carries them directly, while AlgorithmConfig remains available
// for future tunable weights if we choose to expose them.
type CacheAffinityContext struct {
	// TurnIndex is the number of prior turns in this session (0 = first turn).
	TurnIndex int

	// PreviousModel is the model used in the immediately preceding turn.
	// Empty means either first turn or unavailable history.
	PreviousModel string

	// PreviousResponseID is the Response API previous_response_id field.
	// It provides an explicit continuation signal for server-side conversation
	// chains and inline-history requests alike.
	PreviousResponseID string

	// HistoryTokens estimates the token count of prior conversation state,
	// excluding the current user turn.
	HistoryTokens int

	// ContextTokens is the total token count for the current request context.
	ContextTokens int

	// ModelContextWindows maps model name -> ContextWindowSize from ModelParams.
	// Entries that are absent contribute a neutral window-fit score.
	ModelContextWindows map[string]int
}

const (
	// affinityMaxLambda bounds the maximum absolute score adjustment.
	// This keeps cache-affinity as a tie-breaker layered on top of the base
	// hybrid score instead of letting it override strong quality/cost signals.
	affinityMaxLambda = 0.14

	// affinityGapThreshold is the top-2 base-score margin at which ambiguity
	// collapses to zero. Clear base-score separation keeps the hybrid result
	// anchored on the base scorer alone.
	affinityGapThreshold = 0.20

	// historyMassNorm and turnDepthNorm normalize request-time continuity signals
	// into [0,1] before they are combined into W_req.
	historyMassNorm = 4096.0

	turnDepthNorm = 4.0
)

// These constants implement the heuristic from "Pre-Dispatch Cache-Affinity
// Estimator — Final". Keep them explicit so the bounded scoring behavior is
// easy to audit and compare against the design note.
const (
	// W_req weights combine request-level continuation evidence. They should sum
	// to 1.0 so the normalized signal remains easy to reason about.
	wrReuse = 0.45
	wrMass  = 0.30
	wrTurn  = 0.25

	// wrPreviousResponseFloor keeps a small continuation signal for Response API
	// chains that rely on server-side history instead of resending messages.
	wrPreviousResponseFloor = 0.15

	// E_m weights shape per-candidate signed affinity before tanh squashing.
	// same_model terms reward staying on the previous model for continuation-
	// heavy requests; emDiffPenalty pushes against switching when W_req is high;
	// emFit introduces a small context-window fit correction.
	emSameBase    = 0.50
	emSameReuse   = 0.35
	emSameTurn    = 0.15
	emDiffPenalty = 0.35
	emFit         = 0.15

	// C_m weights shape evidence confidence. This gates how much trust to place
	// in A_m before applying the bounded lambda-scaled adjustment.
	cmBase        = 0.15
	cmReuse       = 0.35
	cmMass        = 0.20
	cmTurn        = 0.15
	cmFit         = 0.15
	cmHasPrev     = 0.15
	cmNoPrevScale = 0.60
)

// CacheAffinityResult returns both the per-model score deltas and the
// request-level diagnostics that produced them.
type CacheAffinityResult struct {
	// Adjustments maps model name -> additive score delta.
	Adjustments map[string]float64

	// WReq is the request-level continuation sensitivity in [0,1].
	WReq float64

	// LambdaReq is the ambiguity-gated multiplier applied to all candidates for
	// this request. A zero value leaves the base scores unchanged.
	LambdaReq float64
}

// ComputeCacheAffinityAdjustments applies a bounded pre-dispatch cache-
// affinity bias during hybrid model selection.
//
// The estimator produces a signed affinity effect A_m in (-1,1) and combines
// it with evidence confidence C_m and an ambiguity-gated lambda so that:
//
//	final_score(m) = base_score(m) + lambda_req * C_m * A_m
//
// Production contract:
//   - inputs come from request-time session and candidate metadata
//   - zero adjustment cleanly preserves the base score for first-turn,
//     single-candidate, or unambiguous requests
//   - |adjustment| stays bounded by affinityMaxLambda, keeping cache-affinity
//     as a tie-breaker or gentle bias on close calls
func ComputeCacheAffinityAdjustments(
	ctx *CacheAffinityContext,
	candidates []config.ModelRef,
	baseScores map[string]float64,
) CacheAffinityResult {
	// Fast paths for requests with minimal runtime context or with a single
	// effective candidate.
	if ctx == nil || len(candidates) < 2 {
		return CacheAffinityResult{}
	}

	// hasContinuation is the coarse gate from Step 1. Continuation requests
	// activate the request-level sensitivity calculation below.
	hasContinuation := ctx.TurnIndex > 0 ||
		ctx.HistoryTokens > 0 ||
		ctx.PreviousResponseID != ""
	if !hasContinuation {
		return CacheAffinityResult{}
	}

	result := CacheAffinityResult{}

	// Step 1: request continuation sensitivity W_req.
	contextTokens := math.Max(float64(ctx.ContextTokens), 1)
	reuseRatio := clampF(float64(ctx.HistoryTokens)/contextTokens, 0, 1)
	historyMass := clampF(float64(ctx.HistoryTokens)/historyMassNorm, 0, 1)
	turnDepth := clampF(float64(ctx.TurnIndex)/turnDepthNorm, 0, 1)

	wReq := clampF(wrReuse*reuseRatio+wrMass*historyMass+wrTurn*turnDepth, 0, 1)
	if ctx.PreviousResponseID != "" {
		wReq = math.Max(wReq, wrPreviousResponseFloor)
	}
	result.WReq = wReq

	// Step 4: ambiguity-gated lambda over the current base scores.
	// gap12 is best_score - second_best_score across the candidate set.
	gap12 := computeGap12(baseScores, candidates)
	ambiguity := clampF(1.0-gap12/affinityGapThreshold, 0, 1)
	lambdaReq := affinityMaxLambda * wReq * ambiguity
	result.LambdaReq = lambdaReq

	// A strong base winner keeps the diagnostics while preserving the base
	// scores as-is.
	if lambdaReq == 0 {
		return result
	}

	// Step 2/3/5: per-candidate signed affinity, evidence confidence, then the
	// bounded additive adjustment.
	result.Adjustments = make(map[string]float64, len(candidates))
	hasPrev := boolToFloat64(ctx.PreviousModel != "")

	for _, candidate := range candidates {
		name := candidate.Model

		sameModel := boolToFloat64(ctx.PreviousModel != "" && name == ctx.PreviousModel)
		windowSize := ctx.ModelContextWindows[name]
		fitM := computeFitM(ctx.ContextTokens, windowSize)

		// A_m is a signed effect size that captures the direction and strength of
		// cache-affinity for this candidate.
		eM := emSameBase*sameModel +
			emSameReuse*sameModel*reuseRatio +
			emSameTurn*sameModel*turnDepth -
			emDiffPenalty*(1-sameModel)*wReq +
			emFit*(fitM-0.5)
		aM := math.Tanh(eM)

		// C_m captures how much evidence supports using the affinity signal.
		cRaw := clampF(
			cmBase+
				cmReuse*reuseRatio+
				cmMass*historyMass+
				cmTurn*turnDepth+
				cmFit*fitM+
				cmHasPrev*hasPrev,
			0, 1,
		)
		cM := cRaw
		if ctx.PreviousModel == "" {
			cM = cmNoPrevScale * cRaw
		}

		// cache_adjustment(m) = lambda_req * C_m * A_m
		result.Adjustments[name] = lambdaReq * cM * aM
	}

	logging.Debugf(
		"[CacheAffinity] W_req=%.3f gap12=%.4f ambiguity=%.3f lambda=%.4f "+
			"(turn=%d history=%d ctx=%d prev=%q)",
		wReq, gap12, ambiguity, lambdaReq,
		ctx.TurnIndex, ctx.HistoryTokens, ctx.ContextTokens, ctx.PreviousModel,
	)

	return result
}

// computeFitM implements the per-candidate context-window fit from the design
// note. A missing window size contributes the neutral fit score of 0.5.
func computeFitM(contextTokens, windowSize int) float64 {
	if windowSize <= 0 {
		return 0.5
	}
	w := float64(windowSize)
	c := float64(contextTokens)
	switch {
	case c <= 0.50*w:
		return 1.0
	case c <= 0.75*w:
		return 0.7
	case c <= 1.00*w:
		return 0.3
	default:
		return 0.0
	}
}

// computeGap12 returns best_score - second_best_score over the active
// candidate set. Hybrid selection uses it to decide whether the base result is
// ambiguous enough for cache-affinity to matter.
func computeGap12(baseScores map[string]float64, candidates []config.ModelRef) float64 {
	best := -math.MaxFloat64
	second := -math.MaxFloat64
	for _, c := range candidates {
		s := baseScores[c.Model]
		if s > best {
			second = best
			best = s
		} else if s > second {
			second = s
		}
	}
	if second == -math.MaxFloat64 {
		return 0
	}
	return best - second
}

// boolToFloat64 keeps the estimator formulas close to the math notation from
// the design doc, where indicator terms are written as 0/1 variables.
func boolToFloat64(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

// clampF restricts a float to the closed interval [lo, hi].
func clampF(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(v, hi))
}
