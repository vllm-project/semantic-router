package classification

import (
	"sort"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// firstCopies labels only the first copy of each value, as a model that
// misses the later copies does. Offsets are code points.
func firstCopies(text string, values map[string]string) []modelservice.Span {
	var spans []modelservice.Span
	for value, label := range values {
		index := strings.Index(text, value)
		if index < 0 {
			continue
		}
		start := len([]rune(text[:index]))
		spans = append(spans, modelservice.Span{Label: label, Start: start, End: start + len([]rune(value)), Text: value, Probability: 0.9})
	}
	return spans
}

func byPosition(spans []modelservice.Span) []modelservice.Span {
	sort.SliceStable(spans, func(i, j int) bool {
		return spans[i].Start < spans[j].Start || spans[i].Start == spans[j].Start && spans[i].End < spans[j].End
	})
	return spans
}

// maskSpans replaces each run of covered code points with the label of the
// span that covers its first one.
func maskSpans(text string, spans []modelservice.Span) string {
	runes := []rune(text)
	labels := make([]string, len(runes))
	for _, span := range spans {
		for i := span.Start; i < span.End; i++ {
			if labels[i] == "" {
				labels[i] = span.Label
			}
		}
	}
	var masked strings.Builder
	for i := 0; i < len(runes); {
		if labels[i] == "" {
			masked.WriteRune(runes[i])
			i++
			continue
		}
		masked.WriteString("[" + labels[i] + "]")
		for i < len(runes) && labels[i] != "" {
			i++
		}
	}
	return masked.String()
}

func TestCoverRepeatedSpansCoversEveryCopyOfAValue(t *testing.T) {
	text := "John Smith lives in Boston. Call 555-0142. Yesterday I met John Smith, who said 555-0142 works."
	spans := byPosition(CoverRepeatedSpans(text, firstCopies(text, map[string]string{"John Smith": "PERSON", "555-0142": "PHONE_NUMBER"})))
	var got []string
	for _, span := range spans {
		got = append(got, span.Label+":"+span.Text)
	}
	if want := "PERSON:John Smith | PHONE_NUMBER:555-0142 | PERSON:John Smith | PHONE_NUMBER:555-0142"; strings.Join(got, " | ") != want {
		t.Fatalf("spans in position order:\n got %s\nwant %s", strings.Join(got, " | "), want)
	}
	if masked := maskSpans(text, spans); strings.Contains(masked, "John Smith") || strings.Contains(masked, "555-0142") {
		t.Fatalf("a repeated value must not survive masking: %q", masked)
	}
}

func TestCoverRepeatedSpansCountsCodePoints(t *testing.T) {
	text := "김민지 고객님의 주문이 접수되었습니다. 김민지 고객님께 배송 일정을 안내드립니다."
	spans := byPosition(CoverRepeatedSpans(text, firstCopies(text, map[string]string{"김민지": "PERSON"})))
	if len(spans) != 2 || spans[0].Start != 0 {
		t.Fatalf("want both copies with the first at offset 0, got %+v", spans)
	}
	for _, span := range spans {
		if value := string([]rune(text)[span.Start:span.End]); value != "김민지" {
			t.Fatalf("offsets must index the text's code points, got %q", value)
		}
	}
}

func TestCoverRepeatedSpansCopiesWholeWordsOfTheSameCase(t *testing.T) {
	text := "Will called. Then Will Turner said he will ask Willa and Will_x. Will left."
	spans := byPosition(CoverRepeatedSpans(text, firstCopies(text, map[string]string{"Will": "PERSON"})))
	var starts []int
	for _, span := range spans {
		starts = append(starts, span.Start)
	}
	// "will" differs in case, and "Willa" and "Will_x" are longer words.
	want := []int{0, strings.Index(text, "Will Turner"), strings.LastIndex(text, "Will left")}
	if len(starts) != len(want) || starts[1] != want[1] || starts[2] != want[2] {
		t.Fatalf("got starts %v, want %v", starts, want)
	}
}

// A span over part of a copy is not coverage of the copy: masking only "John"
// of the second "John Smith" would send "Smith" in clear text.
func TestCoverRepeatedSpansCoversTheRestOfAPartlyLabelledCopy(t *testing.T) {
	text := "John Smith met John Smith"
	spans := CoverRepeatedSpans(text, []modelservice.Span{
		{Label: "PERSON", Start: 0, End: 10, Text: "John Smith"},
		{Label: "PERSON", Start: 15, End: 19, Text: "John"},
	})
	if masked := maskSpans(text, spans); masked != "[PERSON] met [PERSON]" {
		t.Fatalf("got %q from %+v", masked, spans)
	}
}

// After "Anna Anna" at the start of "Anna Anna Anna", the next whole-word copy
// starts inside the first one.
func TestCoverRepeatedSpansCoversOverlappingCopies(t *testing.T) {
	text := "Anna Anna Anna"
	spans := CoverRepeatedSpans(text, []modelservice.Span{{Label: "PERSON", Start: 0, End: 9, Text: "Anna Anna"}})
	if masked := maskSpans(text, spans); masked != "[PERSON]" {
		t.Fatalf("got %q from %+v", masked, spans)
	}
}

func TestCoverRepeatedSpansLeavesSingleCharacterValues(t *testing.T) {
	text := "A met A"
	if spans := CoverRepeatedSpans(text, []modelservice.Span{{Label: "PERSON", Start: 0, End: 1, Text: "A"}}); len(spans) != 1 {
		t.Fatalf("a one-character value is not copied: %+v", spans)
	}
}
