package masking

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Span is one detected PII occurrence. Offsets are BYTE offsets into the text
// they were produced from, matching classification.PIIDetection (D2). This
// package deliberately does not import that package, which links cgo (D7).
type Span struct {
	EntityType string
	Start, End int
	Text       string
	Confidence float32
}

// Allocator hands out placeholders. It is request-scoped: the same value gets
// the same placeholder everywhere in one request, and the per-type counter
// keeps incrementing across calls so a later Looper dispatch cannot reuse an
// index for a different value (D5).
type Allocator struct {
	cfg       *config.MaskingPluginConfig
	assigned  map[string]string // entityType + "\x00" + value -> placeholder
	nextIndex map[string]int
}

// NewAllocator constructs an Allocator bound to one decision's masking
// configuration. Callers keep one Allocator for the lifetime of a request
// (D5) so the value-to-placeholder map and the per-type counters accumulate
// across every MaskText call for that request.
func NewAllocator(cfg *config.MaskingPluginConfig) *Allocator {
	return &Allocator{
		cfg:       cfg,
		assigned:  make(map[string]string),
		nextIndex: make(map[string]int),
	}
}

// Placeholder returns the placeholder for one (entityType, value) pair,
// assigning a fresh index from the per-type counter the first time a value is
// seen and reusing it on every subsequent occurrence.
func (a *Allocator) Placeholder(entityType, value string) string {
	key := entityType + "\x00" + value
	if existing, ok := a.assigned[key]; ok {
		return existing
	}
	index := a.nextIndex[entityType]
	a.nextIndex[entityType] = index + 1
	placeholder := strings.ReplaceAll(
		a.cfg.EffectivePlaceholder(entityType),
		config.MaskingPlaceholderIndexToken,
		strconv.Itoa(index),
	)
	a.assigned[key] = placeholder
	return placeholder
}

// filter drops spans whose entity type is excluded by cfg.EntityTypes (an
// include list; empty means every type is masked) or whose confidence is
// below cfg.Threshold (requirement 8).
func (a *Allocator) filter(spans []Span) []Span {
	if len(spans) == 0 {
		return spans
	}
	var allowed map[string]bool
	if len(a.cfg.EntityTypes) > 0 {
		allowed = make(map[string]bool, len(a.cfg.EntityTypes))
		for _, entityType := range a.cfg.EntityTypes {
			allowed[entityType] = true
		}
	}
	filtered := make([]Span, 0, len(spans))
	for _, span := range spans {
		if allowed != nil && !allowed[span.EntityType] {
			continue
		}
		if span.Confidence < a.cfg.Threshold {
			continue
		}
		filtered = append(filtered, span)
	}
	return filtered
}

// MergeSpans sorts ascending and collapses overlaps into the widest covering
// span. Token classifiers routinely emit nested findings (PERSON over "John
// Smith", a narrower label over "John"); splicing both corrupts the text.
func MergeSpans(spans []Span) []Span {
	if len(spans) < 2 {
		return spans
	}
	sorted := append([]Span(nil), spans...)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Start != sorted[j].Start {
			return sorted[i].Start < sorted[j].Start
		}
		return sorted[i].End > sorted[j].End // widest first at equal start
	})
	merged := []Span{sorted[0]}
	for _, candidate := range sorted[1:] {
		last := &merged[len(merged)-1]
		if candidate.Start < last.End {
			if candidate.End > last.End {
				last.End = candidate.End
			}
			continue
		}
		merged = append(merged, candidate)
	}
	return merged
}

// spanReplacement is one resolved splice: the original byte range in the
// source text and the placeholder that replaces it.
type spanReplacement struct {
	start, end  int
	placeholder string
}

// MaskText returns the masked text and the surviving citations. Splicing runs
// descending by start offset so earlier replacements cannot shift later ones.
func MaskText(text string, spans []Span, citations []llmprotocol.Citation, a *Allocator) (string, []llmprotocol.Citation, error) {
	for _, span := range spans {
		// A span the router cannot trust is fail-closed, not skipped (D4).
		if span.Start < 0 || span.End > len(text) || span.Start >= span.End {
			return "", nil, fmt.Errorf("masking: span [%d,%d) outside text of %d bytes", span.Start, span.End, len(text))
		}
	}
	merged := MergeSpans(a.filter(spans))
	if len(merged) == 0 {
		return text, citations, nil
	}

	replacements := make([]spanReplacement, len(merged))
	for i, span := range merged {
		// The value must come from the text, not span.Text: after a merge
		// widens a span, span.Text still holds the narrower original value.
		value := text[span.Start:span.End]
		replacements[i] = spanReplacement{
			start:       span.Start,
			end:         span.End,
			placeholder: a.Placeholder(span.EntityType, value),
		}
	}

	masked := text
	for i := len(replacements) - 1; i >= 0; i-- {
		r := replacements[i]
		masked = masked[:r.start] + r.placeholder + masked[r.end:]
	}

	return masked, adjustCitations(citations, replacements), nil
}

// adjustCitations shifts each citation by the net length delta of masked
// spans that end before it, and drops any citation whose range overlaps a
// masked span (D6): the request validator rejects a citation range that
// falls outside its text block, so a citation into rewritten text is invalid.
func adjustCitations(citations []llmprotocol.Citation, replacements []spanReplacement) []llmprotocol.Citation {
	if len(citations) == 0 {
		return citations
	}
	surviving := make([]llmprotocol.Citation, 0, len(citations))
	for _, citation := range citations {
		start := int(citation.StartIndex)
		end := int(citation.EndIndex)
		delta := 0
		overlaps := false
		for _, r := range replacements {
			if start < r.end && r.start < end {
				overlaps = true
				break
			}
			if r.end <= start {
				delta += len(r.placeholder) - (r.end - r.start)
			}
		}
		if overlaps {
			continue
		}
		citation.StartIndex += int64(delta)
		citation.EndIndex += int64(delta)
		surviving = append(surviving, citation)
	}
	return surviving
}
