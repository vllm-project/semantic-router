package masking

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Span is one detected PII occurrence. Offsets are BYTE offsets into the text
// they were produced from, matching classification.PIIDetection (D2). This
// package deliberately does not import that package, which links cgo (D7).
// Note this differs from llmprotocol.Citation, whose offsets are Unicode
// code points (llmprotocol/types.go) — see citationShift below.
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
	existing  *regexp.Regexp // text already shaped like one of this config's placeholders
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
		existing:  placeholderPattern(cfg),
	}
}

// placeholderPattern matches the placeholders cfg can produce: the default
// [TYPE_n] shape and every configured template with {index} as digits. A
// template with no letter besides {index}, such as "{index}" or "#{index}",
// is left out: its pattern would also match digits inside a real phone
// number or address. It also matches the [vsr:<tag>] marks the vSR Claude
// Code plugin leaves where it hid a value from a tool's output, which reach
// the Router when the plugin and the gateway run together.
func placeholderPattern(cfg *config.MaskingPluginConfig) *regexp.Regexp {
	alternatives := []string{`\[[A-Z][A-Z0-9_]*_\d+\]`, `\[vsr:[a-z]+\]`}
	if cfg != nil {
		entityTypes := make([]string, 0, len(cfg.Placeholders))
		for entityType := range cfg.Placeholders {
			entityTypes = append(entityTypes, entityType)
		}
		sort.Strings(entityTypes)
		for _, entityType := range entityTypes {
			template := cfg.Placeholders[entityType]
			if strings.IndexFunc(strings.ReplaceAll(template, config.MaskingPlaceholderIndexToken, ""), unicode.IsLetter) >= 0 {
				alternatives = append(alternatives, strings.ReplaceAll(regexp.QuoteMeta(template),
					regexp.QuoteMeta(config.MaskingPlaceholderIndexToken), `\d+`))
			}
		}
	}
	return regexp.MustCompile(strings.Join(alternatives, "|"))
}

// maskable is the set of spans MaskText splices into placeholders: the
// configured entity types, without spans inside a placeholder, merged.
func (a *Allocator) maskable(text string, spans []Span) []Span {
	return MergeSpans(a.dropPlaceholderSpans(text, a.filter(spans)))
}

// dropPlaceholderSpans drops spans that lie inside a placeholder already in
// the text. Agents echo placeholders back in later turns (a reply that starts
// "Hi [NAME_2],"), and a detector reading NAME_2 as a name would otherwise
// mask it again into [[NAME_5]], so the model sees its own earlier output
// change under it. A span that reaches past a placeholder, such as an email
// address whose local part is one, is still masked.
func (a *Allocator) dropPlaceholderSpans(text string, spans []Span) []Span {
	found := a.existing.FindAllStringIndex(text, -1)
	if len(found) == 0 {
		return spans
	}
	kept := make([]Span, 0, len(spans))
	for _, span := range spans {
		inside := false
		for _, loc := range found {
			if loc[0] <= span.Start && span.End <= loc[1] {
				inside = true
				break
			}
		}
		if !inside {
			kept = append(kept, span)
		}
	}
	return kept
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
// include list; empty means every type is masked). The pii rule the plugin
// names already dropped the spans below its threshold.
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

// citationShift is one masked range's effect on citation offsets, expressed
// in Unicode code points. llmprotocol.Citation offsets are code-point
// indexes into Content.Text (llmprotocol/types.go), a different unit from
// the byte offsets Span uses to match classifier output (D2), so citation
// math cannot reuse spanReplacement's byte range directly.
type citationShift struct {
	runeStart, runeEnd int
	runeLengthDelta    int
}

// byteOffsetToRuneOffset converts a byte offset into text to the equivalent
// Unicode code-point offset. Mirrors the function of the same name in
// pkg/services/classification_pii_response.go, reimplemented here rather
// than imported because that package links the Rust bindings (D7).
func byteOffsetToRuneOffset(text string, byteOffset int) int {
	if byteOffset <= 0 {
		return 0
	}
	if byteOffset >= len(text) {
		return utf8.RuneCountInString(text)
	}
	return utf8.RuneCountInString(text[:byteOffset])
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
	merged := a.maskable(text, spans)
	if len(merged) == 0 {
		return text, citations, nil
	}

	replacements := make([]spanReplacement, len(merged))
	shifts := make([]citationShift, len(merged))
	for i, span := range merged {
		// The value must come from the text, not span.Text: after a merge
		// widens a span, span.Text still holds the narrower original value.
		value := text[span.Start:span.End]
		placeholder := a.Placeholder(span.EntityType, value)
		replacements[i] = spanReplacement{start: span.Start, end: span.End, placeholder: placeholder}

		runeStart := byteOffsetToRuneOffset(text, span.Start)
		runeEnd := byteOffsetToRuneOffset(text, span.End)
		shifts[i] = citationShift{
			runeStart:       runeStart,
			runeEnd:         runeEnd,
			runeLengthDelta: utf8.RuneCountInString(placeholder) - (runeEnd - runeStart),
		}
	}

	masked := text
	for i := len(replacements) - 1; i >= 0; i-- {
		r := replacements[i]
		masked = masked[:r.start] + r.placeholder + masked[r.end:]
	}

	return masked, adjustCitations(citations, shifts), nil
}

// adjustCitations shifts each citation by the net code-point length delta of
// masked spans that end before it, and drops any citation whose range
// overlaps a masked span (D6): the request validator rejects a citation
// range that falls outside its text block, so a citation into rewritten text
// is invalid.
func adjustCitations(citations []llmprotocol.Citation, shifts []citationShift) []llmprotocol.Citation {
	if len(citations) == 0 {
		return citations
	}
	surviving := make([]llmprotocol.Citation, 0, len(citations))
	for _, citation := range citations {
		start := citation.StartIndex
		end := citation.EndIndex
		var delta int64
		overlaps := false
		for _, shift := range shifts {
			runeStart, runeEnd := int64(shift.runeStart), int64(shift.runeEnd)
			if start < runeEnd && runeStart < end {
				overlaps = true
				break
			}
			if runeEnd <= start {
				delta += int64(shift.runeLengthDelta)
			}
		}
		if overlaps {
			continue
		}
		citation.StartIndex += delta
		citation.EndIndex += delta
		surviving = append(surviving, citation)
	}
	return surviving
}
