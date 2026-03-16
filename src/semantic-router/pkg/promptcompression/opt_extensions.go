package promptcompression

import (
	"fmt"
	"regexp"
	"strings"
)

// opt_extensions.go implements the Claude Code-inspired optimizer plugins:
//
//   - dedup          (PreProcessor)      — drop near-duplicate sentences before scoring
//   - pattern_boost  (AdjustOptimizer)   — multiply composite scores for sentences
//                                          matching error/path/signature regexes
//   - focus_keywords (AdjustOptimizer)   — boost sentences containing user-specified keywords
//   - must_contain   (SelectionOptimizer)— force-keep sentences matching substrings or regexes
//   - role_weight    (AdjustOptimizer)   — per-role composite score multiplier;
//                                          reads ctx.SentenceRoles populated by CompressMessages

// ── dedup ─────────────────────────────────────────────────────────────────────
//
// Removes sentences that are near-duplicates of a later sentence.
// When two sentences exceed the cosine similarity threshold, the earlier one
// is dropped — preserving the more recent instance (recency bias). This is
// the NLP analog of Claude Code's microcompaction: "offload old copies of
// tool results, keep the latest inline."
//
// Uses cosineSimilarityFromTF from textrank.go — no extra dependencies.

type dedupPreProcessor struct {
	threshold float64
}

func (d *dedupPreProcessor) Name() string { return "dedup" }

func (d *dedupPreProcessor) Process(
	sentences []string,
	sentTokens [][]string,
	tfVecs []map[string]float64,
) ([]string, [][]string, []map[string]float64) {
	n := len(sentences)
	if n == 0 {
		return sentences, sentTokens, tfVecs
	}

	drop := make([]bool, n)
	for i := 0; i < n-1; i++ {
		if drop[i] {
			continue
		}
		for j := i + 1; j < n; j++ {
			if cosineSimilarityFromTF(tfVecs[i], tfVecs[j]) >= d.threshold {
				drop[i] = true // keep j (later / more recent)
				break
			}
		}
	}

	outS := make([]string, 0, n)
	outT := make([][]string, 0, n)
	outV := make([]map[string]float64, 0, n)
	for i := range sentences {
		if !drop[i] {
			outS = append(outS, sentences[i])
			outT = append(outT, sentTokens[i])
			outV = append(outV, tfVecs[i])
		}
	}
	return outS, outT, outV
}

// ── pattern_boost ─────────────────────────────────────────────────────────────
//
// Multiplies the composite score of sentences matching any regex in the
// patterns list. Analogous to Claude Code's structured sections 3 and 4
// (errors, file paths with line numbers, function signatures) — content
// that must survive compression for the output to be reconstruction-grade.
//
// Each pattern has an independent multiplier. The first matching pattern
// wins (no stacking). Default patterns are applied when none are configured.

type patternEntry struct {
	re         *regexp.Regexp
	multiplier float64
}

type patternBoostOptimizer struct {
	patterns []patternEntry
}

func (p *patternBoostOptimizer) Name() string { return "pattern_boost" }

func (p *patternBoostOptimizer) Adjust(scored []ScoredSentence, _ OptimizerContext) {
	for i := range scored {
		for _, pe := range p.patterns {
			if pe.re.MatchString(scored[i].Text) {
				scored[i].Composite *= pe.multiplier
				break // first match wins
			}
		}
	}
}

// defaultPatternBoosts are the Claude Code-inspired defaults applied when no
// explicit patterns are supplied in the YAML config.
var defaultPatternBoosts = []struct {
	re   string
	mult float64
}{
	// Errors, exceptions, panics (Claude Code §4: errors encountered)
	{`(?i)(error|exception|panic|fatal|warn)[\s:]`, 1.8},
	// File paths with line numbers: file.go:42, main.py:123 (Claude Code §3: files touched)
	{`\w+\.\w{1,5}:\d+`, 1.6},
	// Function / method signatures (Claude Code §3: exact function signatures)
	{`\bfunc \w+\(|\bdef \w+\(|\bfunction \w+\(`, 1.5},
	// Task markers (Claude Code §7: pending tasks)
	{`\b(TODO|FIXME|HACK|BUG)\b`, 1.4},
}

// ── focus_keywords ────────────────────────────────────────────────────────────
//
// Boosts sentences containing any of the configured keywords.
// This is the NLP equivalent of `/compact focus on X` — the caller declares
// what must survive compression without rewriting the compressor.

type focusKeywordsOptimizer struct {
	keywords []string
	boost    float64
}

func (f *focusKeywordsOptimizer) Name() string { return "focus_keywords" }

func (f *focusKeywordsOptimizer) Adjust(scored []ScoredSentence, _ OptimizerContext) {
	if len(f.keywords) == 0 {
		return
	}
	for i := range scored {
		lower := strings.ToLower(scored[i].Text)
		for _, kw := range f.keywords {
			if strings.Contains(lower, strings.ToLower(kw)) {
				scored[i].Composite *= f.boost
				break
			}
		}
	}
}

// ── must_contain ──────────────────────────────────────────────────────────────
//
// Forces sentences matching any substring or regex into the kept set,
// bypassing the score threshold. Analogous to Claude Code's CLAUDE.md
// "compact instructions" block: "when summarizing, remember X" but as a
// hard guarantee rather than a soft hint to an LLM.

type mustContainOptimizer struct {
	substrings []string
	patterns   []*regexp.Regexp
}

func (m *mustContainOptimizer) Name() string { return "must_contain" }

func (m *mustContainOptimizer) ForceKeep(ctx OptimizerContext) []int {
	var indices []int
	for i, s := range ctx.Sentences {
		for _, sub := range m.substrings {
			if strings.Contains(s, sub) {
				indices = append(indices, i)
				goto next
			}
		}
		for _, re := range m.patterns {
			if re.MatchString(s) {
				indices = append(indices, i)
				goto next
			}
		}
	next:
	}
	return indices
}

// ── role_weight ───────────────────────────────────────────────────────────────
//
// Applies per-role composite score multipliers, modelling Claude Code's
// section 6 ("all user messages verbatim") and aggressive pruning of
// intermediate assistant reasoning.
//
// Role metadata comes from ctx.SentenceRoles, populated by CompressMessages.
// If SentenceRoles is empty (flat text input), this optimizer is a no-op.
//
// Default multipliers:
//   system    1.5   — framing context, important but not as critical as user intent
//   user      2.0   — kept verbatim in Claude Code §6
//   assistant 0.6   — intermediate reasoning is the most droppable content
//   tool      1.0   — TF-IDF already handles tool result relevance

type roleWeightOptimizer struct {
	weights map[string]float64
}

func (r *roleWeightOptimizer) Name() string { return "role_weight" }

func (r *roleWeightOptimizer) Adjust(scored []ScoredSentence, ctx OptimizerContext) {
	if len(ctx.SentenceRoles) == 0 {
		return
	}
	for i := range scored {
		if i >= len(ctx.SentenceRoles) {
			break
		}
		role := ctx.SentenceRoles[i]
		if w, ok := r.weights[role]; ok {
			scored[i].Composite *= w
		}
	}
}

// ── Registration ──────────────────────────────────────────────────────────────

func init() {
	RegisterPreProcessor("dedup", func(params map[string]any) (PreProcessor, error) {
		threshold := 0.95
		if params != nil {
			if t, ok := params["threshold"]; ok {
				threshold = anyToFloat64(t)
			}
		}
		return &dedupPreProcessor{threshold: threshold}, nil
	})

	RegisterAdjust("pattern_boost", func(params map[string]any) (AdjustOptimizer, error) {
		var entries []patternEntry

		if params != nil {
			if raw, ok := params["patterns"]; ok {
				if list, ok := raw.([]any); ok {
					for _, item := range list {
						m, ok := item.(map[string]any)
						if !ok {
							continue
						}
						reStr, _ := m["regex"].(string)
						mult := anyToFloat64(m["multiplier"])
						if mult == 0 {
							mult = 1.5
						}
						re, err := regexp.Compile(reStr)
						if err != nil {
							return nil, fmt.Errorf("pattern_boost: invalid regex %q: %w", reStr, err)
						}
						entries = append(entries, patternEntry{re: re, multiplier: mult})
					}
				}
			}
		}

		// Fall back to defaults when no patterns configured.
		if len(entries) == 0 {
			for _, d := range defaultPatternBoosts {
				re, _ := regexp.Compile(d.re)
				entries = append(entries, patternEntry{re: re, multiplier: d.mult})
			}
		}

		return &patternBoostOptimizer{patterns: entries}, nil
	})

	RegisterAdjust("focus_keywords", func(params map[string]any) (AdjustOptimizer, error) {
		var keywords []string
		boost := 2.0
		if params != nil {
			if raw, ok := params["keywords"]; ok {
				if list, ok := raw.([]any); ok {
					for _, kw := range list {
						if s, ok := kw.(string); ok {
							keywords = append(keywords, s)
						}
					}
				}
			}
			if b, ok := params["boost"]; ok {
				boost = anyToFloat64(b)
			}
		}
		return &focusKeywordsOptimizer{keywords: keywords, boost: boost}, nil
	})

	RegisterAdjust("role_weight", func(params map[string]any) (AdjustOptimizer, error) {
		weights := map[string]float64{
			"system":    1.5,
			"user":      2.0,
			"assistant": 0.6,
			"tool":      1.0,
		}
		if params != nil {
			if raw, ok := params["weights"]; ok {
				if wMap, ok := raw.(map[string]any); ok {
					for role, val := range wMap {
						weights[role] = anyToFloat64(val)
					}
				}
			}
		}
		return &roleWeightOptimizer{weights: weights}, nil
	})

	RegisterSelection("must_contain", func(params map[string]any) (SelectionOptimizer, error) {
		var substrings []string
		var patterns []*regexp.Regexp
		if params != nil {
			if raw, ok := params["substrings"]; ok {
				if list, ok := raw.([]any); ok {
					for _, s := range list {
						if str, ok := s.(string); ok {
							substrings = append(substrings, str)
						}
					}
				}
			}
			if raw, ok := params["patterns"]; ok {
				if list, ok := raw.([]any); ok {
					for _, item := range list {
						if s, ok := item.(string); ok {
							re, err := regexp.Compile(s)
							if err != nil {
								return nil, fmt.Errorf("must_contain: invalid regex %q: %w", s, err)
							}
							patterns = append(patterns, re)
						}
					}
				}
			}
		}
		return &mustContainOptimizer{substrings: substrings, patterns: patterns}, nil
	})
}
