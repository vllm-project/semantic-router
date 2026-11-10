package classification

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// CoverRepeatedSpans adds a span for every other copy of a value the spans
// cover. A model can label a value once and miss its later copies, and
// masking works from positions, so a missed copy would reach the provider in
// clear text. Only exact, word-aligned copies of at least two characters
// count, and only a copy that a span already covers whole is skipped: a span
// over part of a copy still leaves the rest in clear text. A copy takes its
// value's label and probability. Offsets are code points, as in the spans.
func CoverRepeatedSpans(text string, spans []modelservice.Span) []modelservice.Span {
	toByte := modelservice.ByteOffsets(text)
	runes := utf8.RuneCountInString(text)
	covered := make([]byteSpan, 0, len(spans))
	for _, span := range spans {
		if span.Start >= 0 && span.Start < span.End && span.End <= runes {
			covered = append(covered, byteSpan{toByte(span.Start), toByte(span.End)})
		}
	}
	covered = mergeByteSpans(covered)
	searched := make(map[string]bool)
	found := len(spans)
	for i := 0; i < found; i++ {
		source := spans[i]
		if source.Start < 0 || source.Start >= source.End || source.End > runes {
			continue
		}
		value := text[toByte(source.Start):toByte(source.End)]
		if searched[value] || utf8.RuneCountInString(value) < 2 {
			continue
		}
		searched[value] = true
		var added []byteSpan
		lastByte, lastRune := 0, 0
		for from := 0; ; {
			index := strings.Index(text[from:], value)
			if index < 0 {
				break
			}
			start := from + index
			end := start + len(value)
			// Copies can overlap ("Anna Anna" in "Anna Anna Anna"), so the
			// next search starts one rune after this match, not at its end.
			_, size := utf8.DecodeRuneInString(text[start:])
			from = start + size
			if !wordAligned(text, start, end) || containsByteSpan(covered, start, end) {
				continue
			}
			lastRune += utf8.RuneCountInString(text[lastByte:start])
			lastByte = start
			added = append(added, byteSpan{start, end})
			spans = append(spans, modelservice.Span{
				Label:       source.Label,
				Start:       lastRune,
				End:         lastRune + source.End - source.Start,
				Text:        value,
				Probability: source.Probability,
			})
		}
		covered = mergeByteSpans(append(covered, added...))
	}
	if len(spans) > found {
		// Answers list spans in ascending position; the copies keep that.
		sort.SliceStable(spans, func(i, j int) bool {
			return spans[i].Start < spans[j].Start || spans[i].Start == spans[j].Start && spans[i].End < spans[j].End
		})
	}
	return spans
}

type byteSpan struct{ start, end int }

// mergeByteSpans sorts spans and joins the ones that overlap or touch.
func mergeByteSpans(spans []byteSpan) []byteSpan {
	sort.Slice(spans, func(a, b int) bool { return spans[a].start < spans[b].start })
	merged := spans[:0]
	for _, span := range spans {
		if last := len(merged) - 1; last >= 0 && span.start <= merged[last].end {
			merged[last].end = max(merged[last].end, span.end)
			continue
		}
		merged = append(merged, span)
	}
	return merged
}

// containsByteSpan reports whether one of the sorted, disjoint spans
// contains [start, end).
func containsByteSpan(spans []byteSpan, start, end int) bool {
	i := sort.Search(len(spans), func(i int) bool { return spans[i].end > start })
	return i < len(spans) && spans[i].start <= start && spans[i].end >= end
}

// wordAligned reports whether text[start:end] is not part of a longer word.
func wordAligned(text string, start, end int) bool {
	isWord := func(r rune) bool { return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) }
	if before, _ := utf8.DecodeLastRuneInString(text[:start]); start > 0 && isWord(before) {
		return false
	}
	if after, _ := utf8.DecodeRuneInString(text[end:]); end < len(text) && isWord(after) {
		return false
	}
	return true
}
