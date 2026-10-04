package memory

import (
	"math"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// defaultBlockPatterns is empty by default.
// Operators can add custom block_patterns via MemoryReflectionConfig.
var defaultBlockPatterns = []string{}

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
// Uses text-unit Jaccard as a proxy for cosine similarity, so it does not
// need embeddings. Latin words stay one unit; each CJK character is its own.
// The Jaccard bag ignores order, so the same numerals swapped across entities
// score 1. A memory is dropped only when the overlap is high and those
// numeric values stay bound to the same preceding text.
func (g *ReflectionGate) dedup(memories []*RetrieveResult) []*RetrieveResult {
	if len(memories) <= 1 {
		return memories
	}

	kept := make([]*RetrieveResult, 0, len(memories))
	for i, candidate := range memories {
		isDup := false
		for _, existing := range kept {
			if wordJaccard(candidate.Memory.Content, existing.Memory.Content) < g.dedupThreshold {
				continue
			}
			if numericBindingsDiffer(candidate.Memory.Content, existing.Memory.Content) {
				continue
			}
			logging.Debugf("ReflectionGate: dedup memory id=%s (similar to id=%s)",
				candidate.Memory.ID, existing.Memory.ID)
			isDup = true
			break
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

// wordJaccard computes Jaccard similarity on lowercased text units.
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
	units := textUnits(s)
	set := make(map[string]bool, len(units))
	for _, unit := range units {
		set[unit] = true
	}
	return set
}

// cjkNumericRunes are Han and Hangul numeral characters. Magnitude words
// (十百千万, 십백천만) stay here so "一万" and "两万" are numeric bindings,
// not just shared context around 一 and 两.
const cjkNumericRunes = "零〇一二三四五六七八九十百千万亿两兩壹贰貳叁參肆伍陆陸柒捌玖拾佰仟萬億" +
	"영일이삼사오육륙칠팔구십백천만억조"

// numericBindingsDiffer reports whether numeric units are attached to
// different preceding text. Character-set Jaccard is 1 for
// "旅行预算一万美元，机票预算两万美元" and the amounts swapped, because 一 and
// 两 both occur once. The spans before each numeral differ, so the facts stay
// distinct. Texts with no numerals compare equal.
func numericBindingsDiffer(a, b string) bool {
	left := numericBindings(a)
	right := numericBindings(b)
	if len(left) != len(right) {
		return true
	}
	for i := range left {
		if left[i] != right[i] {
			return true
		}
	}
	return false
}

func numericBindings(s string) []string {
	units := textUnits(s)
	bindings := make([]string, 0)
	ctx := make([]string, 0)
	for _, unit := range units {
		if !isNumericUnit(unit) {
			ctx = append(ctx, unit)
			continue
		}
		bindings = append(bindings, strings.Join(ctx, "\x1f")+"\x00"+unit)
		ctx = ctx[:0]
	}
	sort.Strings(bindings)
	return bindings
}

func isNumericUnit(unit string) bool {
	r, size := utf8.DecodeRuneInString(unit)
	if size == 0 {
		return false
	}
	if size == len(unit) && strings.ContainsRune(cjkNumericRunes, r) {
		return true
	}
	for _, digit := range unit {
		if !unicode.IsDigit(digit) {
			return false
		}
	}
	return true
}

// textUnits splits text the same way multilingual signals do. Each Han,
// Hiragana, Katakana, or Hangul character is one unit. A contiguous run of
// other letters or digits is one unit. Punctuation and whitespace separate
// units and are not counted.
func textUnits(s string) []string {
	var units []string
	var latin strings.Builder
	flush := func() {
		if latin.Len() == 0 {
			return
		}
		units = append(units, latin.String())
		latin.Reset()
	}
	for _, r := range strings.ToLower(s) {
		switch {
		case isCJK(r):
			flush()
			units = append(units, string(r))
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			latin.WriteRune(r)
		default:
			flush()
		}
	}
	flush()
	return units
}

func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}

const (
	// latinTokensPerUnit is the English word approximation (words * 1.3).
	latinTokensPerUnit = 1.3
	// cjkTokensPerRune uses the same 1.5 CJK multiplier as
	// promptcompression.CountTokensApprox (conservative vs real tokenizers).
	cjkTokensPerRune = 1.5
)

// estimateTokens estimates injected size for the reflection token budget.
// Latin word units use 1.3 tokens; each Han, Hiragana, Katakana, or Hangul
// character uses 1.5. The total is Ceil'd; CountTokensApprox truncates instead.
// This is a heuristic, not a model tokenizer. Thai is not segmented here.
func estimateTokens(s string) int {
	cjk := 0
	latin := 0
	for _, unit := range textUnits(s) {
		r, size := utf8.DecodeRuneInString(unit)
		if size == len(unit) && isCJK(r) {
			cjk++
			continue
		}
		latin++
	}
	if cjk == 0 && latin == 0 {
		return 0
	}
	return int(math.Ceil(float64(cjk)*cjkTokensPerRune + float64(latin)*latinTokensPerUnit))
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
