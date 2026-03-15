// Package promptcompression — optimizer plugin interfaces and registry.
//
// The optimizer registry decouples scoring logic from the compression pipeline.
// New signals can be added at runtime by calling RegisterScoring, RegisterAdjust,
// RegisterSelection, or RegisterPreProcessor. Built-in optimizers (textrank,
// position, tfidf, novelty, age_decay) are registered in opt_builtin.go.
// Extension optimizers (dedup, pattern_boost, focus_keywords, must_contain,
// role_weight) are registered in opt_extensions.go.
//
// Pipeline phase order:
//  1. PreProcessor  — modifies the sentence list before any scoring (e.g. dedup).
//  2. ScoringOptimizer — returns a [0,1] score per sentence; combined as weighted sum.
//  3. AdjustOptimizer  — multiplicatively modifies composite scores (e.g. pattern boost).
//  4. SelectionOptimizer — forces specific sentences into the kept set unconditionally.
package promptcompression

import (
	"fmt"
	"sync"
)

// OptimizerContext is the read-only data bag passed to every optimizer.
// All slices are parallel: Sentences[i], SentTokens[i], TFVecs[i], TokenCounts[i]
// and (optionally) SentenceRoles[i] all refer to the same sentence.
type OptimizerContext struct {
	Sentences    []string
	SentTokens   [][]string
	TFVecs       []map[string]float64
	TokenCounts  []int
	SentenceRoles []string // optional; "user", "assistant", "system", "tool"
}

// ScoringOptimizer contributes a normalized [0,1] score per sentence.
// Scores from all ScoringOptimizers are combined as a weighted sum to form
// the initial composite score before adjustment.
type ScoringOptimizer interface {
	Name() string
	Score(ctx OptimizerContext) []float64
}

// AdjustOptimizer multiplicatively modifies composite scores in-place after
// all ScoringOptimizers have run. Useful for pattern-based boosts and role
// weighting where a multiplier is more natural than an additive signal.
type AdjustOptimizer interface {
	Name() string
	Adjust(scored []ScoredSentence, ctx OptimizerContext)
}

// SelectionOptimizer returns indices of sentences that must be preserved
// regardless of their composite score. Applied after all scoring and
// adjustment; the budget is still respected on a best-effort basis.
type SelectionOptimizer interface {
	Name() string
	ForceKeep(ctx OptimizerContext) []int
}

// PreProcessor modifies the sentence list (and parallel slices) before any
// scoring. Used for deduplication and other pre-filtering steps. The returned
// slices may be shorter than the input.
type PreProcessor interface {
	Name() string
	Process(
		sentences []string,
		sentTokens [][]string,
		tfVecs []map[string]float64,
	) ([]string, [][]string, []map[string]float64)
}

// ── Registry ──────────────────────────────────────────────────────────────────

var (
	registryMu     sync.RWMutex
	scoringReg     = map[string]func(map[string]any) (ScoringOptimizer, error){}
	adjustReg      = map[string]func(map[string]any) (AdjustOptimizer, error){}
	selectionReg   = map[string]func(map[string]any) (SelectionOptimizer, error){}
	preProcessorReg = map[string]func(map[string]any) (PreProcessor, error){}
)

// RegisterScoring registers a factory for a named ScoringOptimizer.
// Safe for concurrent use; typically called from init().
func RegisterScoring(name string, factory func(map[string]any) (ScoringOptimizer, error)) {
	registryMu.Lock()
	defer registryMu.Unlock()
	scoringReg[name] = factory
}

// RegisterAdjust registers a factory for a named AdjustOptimizer.
func RegisterAdjust(name string, factory func(map[string]any) (AdjustOptimizer, error)) {
	registryMu.Lock()
	defer registryMu.Unlock()
	adjustReg[name] = factory
}

// RegisterSelection registers a factory for a named SelectionOptimizer.
func RegisterSelection(name string, factory func(map[string]any) (SelectionOptimizer, error)) {
	registryMu.Lock()
	defer registryMu.Unlock()
	selectionReg[name] = factory
}

// RegisterPreProcessor registers a factory for a named PreProcessor.
func RegisterPreProcessor(name string, factory func(map[string]any) (PreProcessor, error)) {
	registryMu.Lock()
	defer registryMu.Unlock()
	preProcessorReg[name] = factory
}

func lookupScoring(name string, params map[string]any) (ScoringOptimizer, error) {
	registryMu.RLock()
	factory, ok := scoringReg[name]
	registryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("promptcompression: unknown scoring optimizer %q", name)
	}
	return factory(params)
}

func lookupAdjust(name string, params map[string]any) (AdjustOptimizer, error) {
	registryMu.RLock()
	factory, ok := adjustReg[name]
	registryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("promptcompression: unknown adjust optimizer %q", name)
	}
	return factory(params)
}

func lookupSelection(name string, params map[string]any) (SelectionOptimizer, error) {
	registryMu.RLock()
	factory, ok := selectionReg[name]
	registryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("promptcompression: unknown selection optimizer %q", name)
	}
	return factory(params)
}

func lookupPreProcessor(name string, params map[string]any) (PreProcessor, error) {
	registryMu.RLock()
	factory, ok := preProcessorReg[name]
	registryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("promptcompression: unknown pre-processor %q", name)
	}
	return factory(params)
}
