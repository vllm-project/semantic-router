package promptcompression

import "math"

// opt_builtin.go wraps the four classical NLP scorers already implemented in
// this package (TextRank, Position, TF-IDF, Novelty) as ScoringOptimizer
// plugins, plus a new AgeDecay optimizer that discounts older conversation
// turns — the NLP analog of Claude Code's "hot tail" microcompaction policy.
//
// All five are registered in init() so they are available to any Pipeline
// loaded from a YAML config without extra wiring.

// ── TextRank ──────────────────────────────────────────────────────────────────

type textrankOptimizer struct{}

func (t *textrankOptimizer) Name() string { return "textrank" }

func (t *textrankOptimizer) Score(ctx OptimizerContext) []float64 {
	return NewTextRankScorer().ScoreSentencesWithTF(ctx.TFVecs)
}

// ── Position (Lost in the Middle) ────────────────────────────────────────────

type positionOptimizer struct{ depth float64 }

func (p *positionOptimizer) Name() string { return "position" }

func (p *positionOptimizer) Score(ctx OptimizerContext) []float64 {
	return PositionWeights(len(ctx.Sentences), p.depth)
}

// ── TF-IDF ───────────────────────────────────────────────────────────────────

type tfidfOptimizer struct{}

func (t *tfidfOptimizer) Name() string { return "tfidf" }

func (t *tfidfOptimizer) Score(ctx OptimizerContext) []float64 {
	scorer := NewTFIDFScorer(ctx.SentTokens)
	scores := make([]float64, len(ctx.Sentences))
	for i, tf := range ctx.TFVecs {
		scores[i] = scorer.ScoreSentenceWithTF(tf)
	}
	return scores
}

// ── Novelty (inverse centrality) ─────────────────────────────────────────────

type noveltyOptimizer struct{}

func (n *noveltyOptimizer) Name() string { return "novelty" }

func (n *noveltyOptimizer) Score(ctx OptimizerContext) []float64 {
	scorer := NewNoveltyScorer(ctx.TFVecs)
	scores := make([]float64, len(ctx.Sentences))
	for i, tf := range ctx.TFVecs {
		scores[i] = scorer.ScoreSentence(tf)
	}
	return scores
}

// ── Age Decay ─────────────────────────────────────────────────────────────────
//
// AgeDecay models Claude Code's "hot tail" policy: sentences from older turns
// are down-weighted so that recent context is preferred under compression.
//
// Score(i) = exp(-factor * ages[i])   where ages[i] ∈ {0, 1, 2, ...}
//
// age=0 → newest turn (score=1.0), age=1 → one turn back, etc.
// With the default factor=0.15: age=5 → score≈0.47, age=10 → score≈0.22.
//
// The ages slice is provided via pipeline params or via Config.SentenceAges
// (see CompressMessages which populates this automatically per message turn).
// If ages is empty or shorter than the sentence list, missing entries default
// to age=0 (treated as the current turn).

type ageDecayOptimizer struct {
	ages   []int
	factor float64
}

func (a *ageDecayOptimizer) Name() string { return "age_decay" }

func (a *ageDecayOptimizer) Score(ctx OptimizerContext) []float64 {
	n := len(ctx.Sentences)
	scores := make([]float64, n)
	for i := range scores {
		age := 0
		if i < len(a.ages) {
			age = a.ages[i]
		}
		scores[i] = math.Exp(-a.factor * float64(age))
	}
	return scores
}

// ── Registration ──────────────────────────────────────────────────────────────

func init() {
	RegisterScoring("textrank", func(_ map[string]any) (ScoringOptimizer, error) {
		return &textrankOptimizer{}, nil
	})

	RegisterScoring("position", func(params map[string]any) (ScoringOptimizer, error) {
		depth := 0.5
		if d, ok := params["depth"]; ok {
			depth = anyToFloat64(d)
		}
		return &positionOptimizer{depth: depth}, nil
	})

	RegisterScoring("tfidf", func(_ map[string]any) (ScoringOptimizer, error) {
		return &tfidfOptimizer{}, nil
	})

	RegisterScoring("novelty", func(_ map[string]any) (ScoringOptimizer, error) {
		return &noveltyOptimizer{}, nil
	})

	RegisterScoring("age_decay", func(params map[string]any) (ScoringOptimizer, error) {
		factor := 0.15
		var ages []int
		if params != nil {
			if f, ok := params["factor"]; ok {
				factor = anyToFloat64(f)
			}
			if a, ok := params["ages"]; ok {
				if sl, ok := a.([]any); ok {
					ages = make([]int, len(sl))
					for i, v := range sl {
						ages[i] = int(anyToFloat64(v))
					}
				}
			}
		}
		return &ageDecayOptimizer{ages: ages, factor: factor}, nil
	})
}

// anyToFloat64 safely coerces YAML-decoded interface{} values to float64.
// YAML v3 unmarshals numbers as int, float64, or string depending on format.
func anyToFloat64(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int:
		return float64(x)
	case int32:
		return float64(x)
	case int64:
		return float64(x)
	case uint:
		return float64(x)
	case uint64:
		return float64(x)
	}
	return 0
}
