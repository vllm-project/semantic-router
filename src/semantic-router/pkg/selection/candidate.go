package selection

import (
	"math"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// CandidateID identifies the complete configured choice, not just its model.
// It survives filtering, sorting and composition. Pointer-valued controls are
// compared by value, so copies and restored session candidates keep their identity.
// This is internal routing identity, not a new user-facing model identifier.
type CandidateID struct {
	ref       config.ModelRef
	reasoning int8
}

func CandidateIdentity(ref config.ModelRef) CandidateID {
	id := CandidateID{ref: ref}
	if ref.UseReasoning != nil {
		id.reasoning = 1
		if *ref.UseReasoning {
			id.reasoning = 2
		}
	}
	id.ref.UseReasoning = nil
	return id
}

// CandidateScore retains the exact subject of a value. Evidence describes only
// the benchmark contribution; it must not be attached to learned-only scores.
type CandidateScore struct {
	Candidate config.ModelRef
	Score     float64
	Evidence  *modelcatalog.IndexResult
}

type CandidateScores []CandidateScore

type ScoreDirection uint8

const (
	HigherIsBetter ScoreDirection = iota
	LowerIsBetter
)

// ponytail: linear lookup over small routing pools; index CandidateID if profiling warrants it.
func (scores CandidateScores) Get(ref config.ModelRef) (float64, bool) {
	id := CandidateIdentity(ref)
	for _, row := range scores {
		if CandidateIdentity(row.Candidate) == id {
			return row.Score, true
		}
	}
	return 0, false
}

// Best preserves input order for unresolved ties. This is a winner reduction,
// not a sort comparator: incomparable evidence must not become an ordering key.
func (scores CandidateScores) Best(direction ScoreDirection, coverage bool) int {
	best := -1
	for i, row := range scores {
		if math.IsNaN(row.Score) || math.IsInf(row.Score, 0) {
			continue
		}
		if best < 0 {
			best = i
			continue
		}
		left, right := candidateRank{score: row.Score}, candidateRank{score: scores[best].Score}
		if direction == LowerIsBetter {
			left.score, right.score = -left.score, -right.score
		}
		if coverage {
			left.evidence, right.evidence = row.Evidence, scores[best].Evidence
		}
		if left.Compare(right) > 0 {
			best = i
		}
	}
	return best
}

// Diagnostics is a presentation projection. Never read it back to rank candidates.
func (scores CandidateScores) Diagnostics() map[string]float64 {
	refs := make([]config.ModelRef, len(scores))
	for i := range scores {
		refs[i] = scores[i].Candidate
	}
	out := make(map[string]float64, len(scores))
	for i, row := range scores {
		out[candidateScoreKey(refs, i)] = row.Score
	}
	return out
}

// ScoresFor is the compatibility boundary for older/model-level scorers.
// Once typed scores exist, absent entries stay absent; formatted diagnostic keys
// are never parsed or used as a fallback. Model-level priors are explicitly
// expanded to individual candidates without asserting effort-specific evidence.
func (result *SelectionResult) ScoresFor(refs []config.ModelRef) CandidateScores {
	if result == nil {
		return nil
	}
	if result.CandidateScores != nil {
		out := make(CandidateScores, 0, len(refs))
		allowed := make(map[CandidateID]bool, len(refs))
		for _, ref := range refs {
			allowed[CandidateIdentity(ref)] = true
		}
		for _, row := range result.CandidateScores {
			if allowed[CandidateIdentity(row.Candidate)] {
				out = append(out, row)
			}
		}
		return out
	}
	out := make(CandidateScores, 0, len(refs))
	for _, ref := range refs {
		if score, ok := result.AllScores[ref.Model]; ok {
			out = append(out, CandidateScore{Candidate: ref, Score: score})
		} else if result.SelectedCandidate != nil && CandidateIdentity(*result.SelectedCandidate) == CandidateIdentity(ref) {
			out = append(out, CandidateScore{Candidate: ref, Score: result.Score})
		}
	}
	return out
}

func (result *SelectionResult) WithCandidate(ref config.ModelRef) *SelectionResult {
	out := *result
	if ref.UseReasoning != nil {
		enabled := *ref.UseReasoning
		ref.UseReasoning = &enabled
	}
	out.SelectedCandidate = &ref
	out.SelectedModel, out.LoRAName = ref.Model, ref.LoRAName
	return &out
}

func (result *SelectionResult) WithScores(scores CandidateScores) *SelectionResult {
	out := *result
	out.CandidateScores = append(CandidateScores(nil), scores...)
	out.AllScores = scores.Diagnostics()
	return &out
}

// CurrentSessionCandidate preserves a recorded exact choice across requests.
// Older model-only sessions can reuse the base winner for that same model;
// they cannot guess between other ambiguous candidates.
func CurrentSessionCandidate(ctx *SelectionContext, base *SelectionResult, model string) *config.ModelRef {
	if ctx == nil || base == nil {
		return nil
	}
	if ctx.AgenticSession != nil && ctx.AgenticSession.PreviousCandidate != nil {
		previous := ctx.AgenticSession.PreviousCandidate
		if previous.Model == model || previous.LoRAName == model {
			for _, ref := range ctx.CandidateModels {
				if CandidateIdentity(ref) == CandidateIdentity(*previous) {
					return (&SelectionResult{}).WithCandidate(ref).SelectedCandidate
				}
			}
			return nil
		}
	}
	return CandidateForModel(ctx.CandidateModels, model, base.SelectedCandidate)
}

// CandidateForModel resolves legacy/session model identity only when unique, or
// preserves a known exact candidate. It never picks the first ambiguous effort.
func CandidateForModel(refs []config.ModelRef, model string, preferred *config.ModelRef) *config.ModelRef {
	if model == "" {
		return nil
	}
	if preferred != nil && (preferred.Model == model || preferred.LoRAName == model) {
		for _, ref := range refs {
			if CandidateIdentity(ref) == CandidateIdentity(*preferred) {
				return (&SelectionResult{}).WithCandidate(ref).SelectedCandidate
			}
		}
	}
	// A concrete model name takes precedence over another model's LoRA alias.
	exactModel := false
	for _, ref := range refs {
		exactModel = exactModel || ref.Model == model
	}
	var found *config.ModelRef
	for _, ref := range refs {
		if exactModel && ref.Model != model {
			continue
		}
		if ref.Model != model && (ref.LoRAName == "" || ref.LoRAName != model) {
			continue
		}
		if found != nil && CandidateIdentity(*found) != CandidateIdentity(ref) {
			return nil
		}
		found = (&SelectionResult{}).WithCandidate(ref).SelectedCandidate
	}
	return found
}
