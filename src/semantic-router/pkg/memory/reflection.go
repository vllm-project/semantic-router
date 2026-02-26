package memory

import (
	"math"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Default block patterns are intentionally empty. The primary adversarial
// content gate is SR's jailbreak classifier which runs on the request path;
// only requests that pass the classifier reach the LLM and produce memories.
// The write-path fallback lives in sanitize.go for when the classifier is
// disabled. Users can add custom patterns via config if needed.
var defaultBlockPatterns []string

// ReflectionGate filters retrieved memories before injection using heuristic
// rules: recency decay, redundancy dedup, and token budget enforcement.
// No LLM calls -- sub-millisecond overhead.
//
// Adversarial content blocking is handled upstream: SR's jailbreak classifier
// gates the write path (only clean requests produce memories), and sanitize.go
// provides a regex fallback. Custom block patterns can still be added via config.
//
// Inspired by RMM (ACL 2025) retrospective reflection, adapted for
// infrastructure-layer operation where latency must be minimal.
type ReflectionGate struct {
	maxTokens      int
	decayHalfLife  float64 // days
	dedupThreshold float32
	blockPatterns  []*regexp.Regexp
}

// newHeuristicFilter is the MemoryFilterFactory registered as "heuristic".
func newHeuristicFilter(global config.MemoryReflectionConfig, perDecision *config.MemoryReflectionConfig) MemoryFilter {
	return NewReflectionGate(global, perDecision)
}

// NewReflectionGate creates a gate from the global config, optionally
// overridden by a per-decision plugin config.
// Prefer NewMemoryFilter() in production code -- it selects the algorithm
// from config and delegates to the appropriate factory.
func NewReflectionGate(global config.MemoryReflectionConfig, perDecision *config.MemoryReflectionConfig) *ReflectionGate {
	cfg := resolveReflectionConfig(global, perDecision)

	if !cfg.ReflectionEnabled() {
		return nil
	}

	maxTokens := cfg.MaxInjectTokens
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	decayDays := cfg.RecencyDecayDays
	if decayDays <= 0 {
		decayDays = 30
	}

	dedupThresh := cfg.DedupThreshold
	if dedupThresh <= 0 {
		dedupThresh = 0.90
	}

	patterns := buildBlockPatterns(cfg.BlockPatterns)

	return &ReflectionGate{
		maxTokens:      maxTokens,
		decayHalfLife:  float64(decayDays),
		dedupThreshold: dedupThresh,
		blockPatterns:  patterns,
	}
}

// Filter applies all heuristic checks and returns the memories that pass.
// The returned slice is a subset of the input, re-scored and trimmed to
// the token budget.
func (g *ReflectionGate) Filter(memories []*RetrieveResult) []*RetrieveResult {
	if g == nil || len(memories) == 0 {
		return memories
	}

	now := time.Now()
	original := len(memories)

	// Step 1: Block adversarial content
	memories = g.filterBlocked(memories)

	// Step 2: Apply recency decay to scores
	g.applyRecencyDecay(memories, now)

	// Step 3: Re-sort by adjusted score (descending)
	sort.Slice(memories, func(i, j int) bool {
		return memories[i].Score > memories[j].Score
	})

	// Step 4: Deduplicate near-identical memories
	memories = g.dedup(memories)

	// Step 5: Enforce token budget
	memories = g.enforceTokenBudget(memories)

	if len(memories) < original {
		logging.Debugf("ReflectionGate: %d→%d memories (blocked=%d, dedup+budget trimmed)",
			original, len(memories), original-len(memories))
	}

	return memories
}

// filterBlocked removes memories whose content matches any block pattern.
func (g *ReflectionGate) filterBlocked(memories []*RetrieveResult) []*RetrieveResult {
	kept := make([]*RetrieveResult, 0, len(memories))
	for _, m := range memories {
		if m.Memory == nil {
			continue
		}
		blocked := false
		for _, pat := range g.blockPatterns {
			if pat.MatchString(m.Memory.Content) {
				logging.Debugf("ReflectionGate: blocked memory id=%s (pattern=%s)", m.Memory.ID, pat.String())
				blocked = true
				break
			}
		}
		if !blocked {
			kept = append(kept, m)
		}
	}
	return kept
}

// applyRecencyDecay multiplies each memory's score by an exponential decay
// factor based on age: factor = exp(-0.693 * age_days / halfLife).
// At halfLife days old the factor is 0.5; at 0 days it's 1.0.
func (g *ReflectionGate) applyRecencyDecay(memories []*RetrieveResult, now time.Time) {
	ln2 := math.Ln2 // 0.693...
	for _, m := range memories {
		if m.Memory == nil || m.Memory.CreatedAt.IsZero() {
			continue
		}
		ageDays := now.Sub(m.Memory.CreatedAt).Hours() / 24.0
		if ageDays < 0 {
			ageDays = 0
		}
		factor := math.Exp(-ln2 * ageDays / g.decayHalfLife)
		m.Score = float32(float64(m.Score) * factor)
	}
}

// dedup removes memories that are near-duplicates of a higher-scored memory.
// Uses simple content similarity (Jaccard on word sets) as a proxy for
// cosine similarity -- avoids needing embeddings at this stage.
func (g *ReflectionGate) dedup(memories []*RetrieveResult) []*RetrieveResult {
	if len(memories) <= 1 {
		return memories
	}

	kept := make([]*RetrieveResult, 0, len(memories))
	for i, candidate := range memories {
		isDup := false
		for _, existing := range kept {
			if wordJaccard(candidate.Memory.Content, existing.Memory.Content) >= g.dedupThreshold {
				logging.Debugf("ReflectionGate: dedup memory id=%s (similar to id=%s)",
					candidate.Memory.ID, existing.Memory.ID)
				isDup = true
				break
			}
		}
		if !isDup {
			kept = append(kept, memories[i])
		}
	}
	return kept
}

// enforceTokenBudget keeps memories in score-descending order until the
// cumulative estimated token count exceeds the budget.
func (g *ReflectionGate) enforceTokenBudget(memories []*RetrieveResult) []*RetrieveResult {
	var total int
	for i, m := range memories {
		tokens := estimateTokens(m.Memory.Content)
		if total+tokens > g.maxTokens && i > 0 {
			logging.Debugf("ReflectionGate: token budget %d reached at %d/%d memories", g.maxTokens, i, len(memories))
			return memories[:i]
		}
		total += tokens
	}
	return memories
}

// wordJaccard computes Jaccard similarity on lowercased word sets.
func wordJaccard(a, b string) float32 {
	setA := wordSet(a)
	setB := wordSet(b)

	if len(setA) == 0 && len(setB) == 0 {
		return 1.0
	}

	intersection := 0
	for w := range setA {
		if setB[w] {
			intersection++
		}
	}
	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0
	}
	return float32(intersection) / float32(union)
}

func wordSet(s string) map[string]bool {
	words := strings.Fields(strings.ToLower(s))
	set := make(map[string]bool, len(words))
	for _, w := range words {
		set[w] = true
	}
	return set
}

// estimateTokens gives a rough token count (words * 1.3, the common
// English approximation). Good enough for budget enforcement.
func estimateTokens(s string) int {
	words := len(strings.Fields(s))
	return int(math.Ceil(float64(words) * 1.3))
}

// resolveReflectionConfig merges per-decision overrides into global config.
func resolveReflectionConfig(global config.MemoryReflectionConfig, perDecision *config.MemoryReflectionConfig) config.MemoryReflectionConfig {
	if perDecision == nil {
		return global
	}
	merged := global
	if perDecision.Enabled != nil {
		merged.Enabled = perDecision.Enabled
	}
	if perDecision.Algorithm != "" {
		merged.Algorithm = perDecision.Algorithm
	}
	if perDecision.MaxInjectTokens > 0 {
		merged.MaxInjectTokens = perDecision.MaxInjectTokens
	}
	if perDecision.RecencyDecayDays > 0 {
		merged.RecencyDecayDays = perDecision.RecencyDecayDays
	}
	if perDecision.DedupThreshold > 0 {
		merged.DedupThreshold = perDecision.DedupThreshold
	}
	if len(perDecision.BlockPatterns) > 0 {
		merged.BlockPatterns = perDecision.BlockPatterns
	}
	return merged
}

func buildBlockPatterns(userPatterns []string) []*regexp.Regexp {
	all := make([]string, 0, len(defaultBlockPatterns)+len(userPatterns))
	all = append(all, defaultBlockPatterns...)
	all = append(all, userPatterns...)
	compiled := make([]*regexp.Regexp, 0, len(all))
	for _, p := range all {
		re, err := regexp.Compile(p)
		if err != nil {
			logging.Warnf("ReflectionGate: invalid block pattern %q: %v", p, err)
			continue
		}
		compiled = append(compiled, re)
	}
	return compiled
}
