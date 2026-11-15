package masking

import (
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
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
	// shapes are the placeholders this config writes, one per pii label and
	// rule, and the vSR plugins' marks. shaped holds the entity types listed.
	shapes []placeholderShape
	shaped map[string]bool
	rules  map[string]bool
	// issued holds the placeholders this allocator handed out, and reserved
	// the placeholder-shaped text the request held before this allocator
	// issued it, which a new value never gets.
	issued, reserved map[string]bool
}

// placeholderShape is what one entity type's placeholders look like. The vSR
// plugins leave [vsr:<tag>] where they hid a value from a tool's output, a
// tag of their own or the name of a listener rule, and those marks reach the
// Router when a plugin and the gateway run together; their shape has no
// entity type. A shape is lettered when its template has a letter besides
// {index}: "#{index}" is not, since its digits could be part of a phone
// number.
type placeholderShape struct {
	pattern    *regexp.Regexp
	entityType string
	lettered   bool
}

// NewAllocator constructs an Allocator bound to one decision's masking
// configuration. Callers keep one Allocator for the lifetime of a request
// (D5) so the value-to-placeholder map and the per-type counters accumulate
// across every MaskText call for that request.
func NewAllocator(cfg *config.MaskingPluginConfig) *Allocator {
	a := &Allocator{
		cfg:       cfg,
		assigned:  make(map[string]string),
		nextIndex: make(map[string]int),
		shapes:    []placeholderShape{{pattern: shapePattern(`\[vsr:([A-Za-z0-9_.-]+)\]`), lettered: true}},
		shaped:    make(map[string]bool),
		rules:     make(map[string]bool),
		issued:    make(map[string]bool),
		reserved:  make(map[string]bool),
	}
	entityTypes := config.MaskingEntityTypes()
	if cfg != nil {
		for _, rule := range cfg.RuleNames() {
			a.rules[rule] = true
			entityTypes = append(entityTypes, rule)
		}
	}
	for _, entityType := range entityTypes {
		a.addShape(entityType)
	}
	return a
}

// addShape lists an entity type's placeholders. Placeholder also calls it, so
// a label the detector answers outside the pii preset is recognized once it
// was issued.
func (a *Allocator) addShape(entityType string) {
	if a.shaped[entityType] {
		return
	}
	a.shaped[entityType] = true
	template := a.cfg.EffectivePlaceholder(entityType)
	a.shapes = append(a.shapes, placeholderShape{
		pattern:    shapePattern(templatePattern(template)),
		entityType: entityType,
		lettered:   hasLetter(strings.ReplaceAll(template, config.MaskingPlaceholderIndexToken, "")),
	})
}

// compiledShapes holds each placeholder pattern compiled once per process,
// since every request builds its own Allocator.
var compiledShapes sync.Map

func shapePattern(pattern string) *regexp.Regexp {
	if compiled, ok := compiledShapes.Load(pattern); ok {
		return compiled.(*regexp.Regexp)
	}
	compiled, _ := compiledShapes.LoadOrStore(pattern, regexp.MustCompile(pattern))
	return compiled.(*regexp.Regexp)
}

// indexPattern is an index Placeholder can write: a counter in decimal, so
// no leading zero, and far below a billion values in one request. The bound
// keeps a template such as "{index}x" from reading a long run of digits again
// at every digit.
const indexPattern = `(0|[1-9][0-9]{0,8})`

// templatePattern matches what a placeholder template produces, any index,
// with the index as its one capture group.
func templatePattern(template string) string {
	return strings.ReplaceAll(regexp.QuoteMeta(template), regexp.QuoteMeta(config.MaskingPlaceholderIndexToken), indexPattern)
}

// occurrences finds every match of a placeholder pattern, those that overlap
// included: with the template "__NAME{index}__", "__NAME1__NAME2__" holds two,
// and with "X0X{index}X", "X0X0X1X" does too. The next search starts a byte
// after a match starts,
// except after the template "{index}" alone, since a digit run is one index
// and its tail is not another.
func occurrences(pattern *regexp.Regexp, text string) [][]int {
	var found [][]int
	for at := 0; at < len(text); {
		loc := pattern.FindStringSubmatchIndex(text[at:])
		if loc == nil {
			break
		}
		for i := range loc {
			loc[i] += at
		}
		found = append(found, loc)
		at = loc[0] + 1
		if loc[2] == loc[0] && loc[3] == loc[1] {
			at = loc[3]
		}
	}
	return found
}

// maskable is the set of spans MaskText splices into placeholders: the
// configured entity types, without spans inside a placeholder, merged. Each
// text passes here in request order, so a placeholder the request holds
// before this allocator issued it is reserved first: a client may have typed
// it, and a new value must not take its meaning.
func (a *Allocator) maskable(text string, spans []Span) []Span {
	a.reserveUnissued(text)
	return MergeSpans(a.dropPlaceholderSpans(text, a.filter(spans)))
}

// anyDefaultShape is the default [TYPE_n] of any type. Reservation reads it
// as well as the shapes listed, since a label the detector answers outside
// the pii preset gets its shape only when its first value is issued, and its
// text in an earlier block must be reserved by then.
var anyDefaultShape = regexp.MustCompile(`\[[A-Z][A-Z0-9_]*_` + indexPattern + `\]`)

func (a *Allocator) reserveUnissued(text string) {
	patterns := []*regexp.Regexp{anyDefaultShape}
	for _, shape := range a.shapes {
		if shape.entityType != "" {
			patterns = append(patterns, shape.pattern)
		}
	}
	for _, pattern := range patterns {
		for _, loc := range occurrences(pattern, text) {
			if found := text[loc[0]:loc[1]]; !a.issued[found] {
				a.reserved[found] = true
			}
		}
	}
}

// dropPlaceholderSpans drops spans that lie inside a placeholder already in
// the text. Agents echo placeholders back in later turns (a reply that starts
// "Hi [NAME_2],"), and a detector reading NAME_2 as a name would otherwise
// mask it again into [[NAME_5]], so the model sees its own earlier output
// change under it. A span that reaches past a placeholder, such as an email
// address whose local part is one, is still masked. Only the shapes this
// config writes count, so [X_104233] or [PERSON_ALICE_0] is not one. A
// placeholder this allocator issued exempts any span inside it, whichever
// type found it: the history an echo comes from carries its value first. Any
// other one is text a client can type, so it exempts less. A rule's span is
// dropped there only inside the rule's own shape and only when it misses the
// index, since a rule matches the value itself and [EMPLOYEE_ID_104233] or,
// with the template "[E-{index}]", [E-104233] wraps one. A detector's span is
// dropped there only when it takes in a letter of the shape's own text, as a
// misread of NAME_2 or vsr:name does, while a wrapped value such as
// [vsr:4111111111111111] has none.
func (a *Allocator) dropPlaceholderSpans(text string, spans []Span) []Span {
	found := make([][]placeholderAt, len(a.shapes))
	seen := make([]bool, len(a.shapes))
	exempt := func(span Span) bool {
		for i, shape := range a.shapes {
			if !seen[i] {
				found[i], seen[i] = a.placeholdersIn(shape.pattern, text), true
			}
			places := found[i]
			// Walk back from the last placeholder that starts at or before the
			// span, while one of them still reaches the span's end.
			for at := sort.Search(len(places), func(j int) bool { return places[j].loc[0] > span.Start }) - 1; at >= 0 && places[at].reach >= span.End; at-- {
				loc := places[at].loc
				switch {
				case span.End > loc[1]: // ends before the span does
				case places[at].issued:
					return true
				case a.rules[span.EntityType]:
					if shape.entityType == span.EntityType && (span.End <= loc[2] || loc[3] <= span.Start) {
						return true
					}
				case shape.lettered && (letterIn(text, max(span.Start, loc[0]), min(span.End, loc[2])) || letterIn(text, max(span.Start, loc[3]), min(span.End, loc[1]))):
					return true
				}
			}
		}
		return false
	}
	kept := make([]Span, 0, len(spans))
	for _, span := range spans {
		if !exempt(span) {
			kept = append(kept, span)
		}
	}
	return kept
}

// placeholderAt is one placeholder in a text: where it and its index lie,
// whether this allocator issued it, and the furthest any placeholder up to
// it reaches, which bounds the walk back from a span.
type placeholderAt struct {
	loc    []int
	issued bool
	reach  int
}

func (a *Allocator) placeholdersIn(pattern *regexp.Regexp, text string) []placeholderAt {
	var places []placeholderAt
	reach := 0
	for _, loc := range occurrences(pattern, text) {
		reach = max(reach, loc[1])
		places = append(places, placeholderAt{loc: loc, issued: a.issued[text[loc[0]:loc[1]]], reach: reach})
	}
	return places
}

func hasLetter(text string) bool {
	return strings.IndexFunc(text, unicode.IsLetter) >= 0
}

// letterIn reports a letter in text[from:to], where a range that ends before
// it starts holds none.
func letterIn(text string, from, to int) bool {
	return from < to && hasLetter(text[from:to])
}

// Placeholder returns the placeholder for one (entityType, value) pair,
// assigning a fresh index from the per-type counter the first time a value is
// seen and reusing it on every subsequent occurrence.
func (a *Allocator) Placeholder(entityType, value string) string {
	key := entityType + "\x00" + value
	if existing, ok := a.assigned[key]; ok {
		return existing
	}
	for {
		index := a.nextIndex[entityType]
		a.nextIndex[entityType] = index + 1
		placeholder := strings.ReplaceAll(
			a.cfg.EffectivePlaceholder(entityType),
			config.MaskingPlaceholderIndexToken,
			strconv.Itoa(index),
		)
		if a.reserved[placeholder] {
			continue
		}
		a.assigned[key] = placeholder
		a.issued[placeholder] = true
		a.addShape(entityType)
		return placeholder
	}
}

// filter drops spans whose entity type is excluded by cfg.EntityTypes (an
// include list of pii labels; empty means every type is masked). A span of
// one of cfg.Rules is always kept. The pii rule the plugin names already
// dropped the spans below its threshold.
func (a *Allocator) filter(spans []Span) []Span {
	if len(spans) == 0 {
		return spans
	}
	var allowed map[string]bool
	if len(a.cfg.EntityTypes) > 0 {
		allowed = make(map[string]bool, len(a.cfg.EntityTypes)+len(a.cfg.Rules))
		for _, entityType := range append(slices.Clone(a.cfg.EntityTypes), a.cfg.RuleNames()...) {
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

// citationShift is one masked range's effect on citation offsets, expressed
// in Unicode code points. llmprotocol.Citation offsets are code-point
// indexes into Content.Text (llmprotocol/types.go), a different unit from
// the byte offsets Span uses to match classifier output (D2), so citation
// math cannot reuse a span's byte range directly.
type citationShift struct {
	runeStart, runeEnd int
	runeLengthDelta    int
}

func runeBoundary(text string, at int) bool {
	return at == len(text) || utf8.RuneStart(text[at])
}

// MaskText returns the masked text and the surviving citations. The merged
// spans are sorted and disjoint, so one pass in offset order writes the
// masked text and counts code points as it goes: splicing one span at a time
// and counting each offset from the start copied and read the whole text once
// per span.
func MaskText(text string, spans []Span, citations []llmprotocol.Citation, a *Allocator) (string, []llmprotocol.Citation, error) {
	for _, span := range spans {
		// A span the router cannot trust is fail-closed, not skipped (D4).
		if span.Start < 0 || span.End > len(text) || span.Start >= span.End {
			return "", nil, fmt.Errorf("masking: span [%d,%d) outside text of %d bytes", span.Start, span.End, len(text))
		}
		// One that cuts a character would leave part of it beside the
		// placeholder, and the code points counted for citations would not
		// add up across the cut.
		if !runeBoundary(text, span.Start) || !runeBoundary(text, span.End) {
			return "", nil, fmt.Errorf("masking: span [%d,%d) splits a character", span.Start, span.End)
		}
	}
	merged := a.maskable(text, spans)
	if len(merged) == 0 {
		return text, citations, nil
	}

	var masked strings.Builder
	masked.Grow(len(text))
	shifts := make([]citationShift, len(merged))
	at, runeAt := 0, 0
	for i, span := range merged {
		// The value must come from the text, not span.Text: after a merge
		// widens a span, span.Text still holds the narrower original value.
		value := text[span.Start:span.End]
		placeholder := a.Placeholder(span.EntityType, value)
		runeStart := runeAt + utf8.RuneCountInString(text[at:span.Start])
		runeEnd := runeStart + utf8.RuneCountInString(value)
		shifts[i] = citationShift{
			runeStart:       runeStart,
			runeEnd:         runeEnd,
			runeLengthDelta: utf8.RuneCountInString(placeholder) - (runeEnd - runeStart),
		}
		masked.WriteString(text[at:span.Start])
		masked.WriteString(placeholder)
		at, runeAt = span.End, runeEnd
	}
	masked.WriteString(text[at:])

	return masked.String(), adjustCitations(citations, shifts), nil
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
	// The shifts are sorted and disjoint, so those a citation follows are a
	// prefix found by search, and only the next one can overlap it.
	before := make([]int64, len(shifts)+1)
	for i, shift := range shifts {
		before[i+1] = before[i] + int64(shift.runeLengthDelta)
	}
	surviving := make([]llmprotocol.Citation, 0, len(citations))
	for _, citation := range citations {
		next := sort.Search(len(shifts), func(i int) bool { return int64(shifts[i].runeEnd) > citation.StartIndex })
		if next < len(shifts) && int64(shifts[next].runeStart) < citation.EndIndex {
			continue
		}
		citation.StartIndex += before[next]
		citation.EndIndex += before[next]
		surviving = append(surviving, citation)
	}
	return surviving
}
