package masking

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// byteRange returns the byte offsets of substr within text, matching how a
// real classifier reports spans (D2: byte offsets, not rune offsets).
func byteRange(t *testing.T, text, substr string) (int, int) {
	t.Helper()
	start := strings.Index(text, substr)
	if start < 0 {
		t.Fatalf("substring %q not found in %q", substr, text)
	}
	return start, start + len(substr)
}

func defaultCfg() *config.MaskingPluginConfig {
	return &config.MaskingPluginConfig{}
}

// Case: Single span.
func TestMaskText_SingleSpan(t *testing.T) {
	text := "mail alice@x.com"
	start, end := byteRange(t, text, "alice@x.com")
	spans := []Span{{EntityType: "EMAIL_ADDRESS", Start: start, End: end, Confidence: 1.0}}

	got, citations, err := MaskText(text, spans, nil, NewAllocator(defaultCfg()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "mail [EMAIL_ADDRESS_0]"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	if len(citations) != 0 {
		t.Fatalf("expected no citations, got %v", citations)
	}
}

// Case: Repeat value.
func TestMaskText_RepeatValue(t *testing.T) {
	text := "alice@x.com wrote to alice@x.com"
	first := strings.Index(text, "alice@x.com")
	second := strings.LastIndex(text, "alice@x.com")
	spans := []Span{
		{EntityType: "EMAIL_ADDRESS", Start: first, End: first + len("alice@x.com"), Confidence: 1.0},
		{EntityType: "EMAIL_ADDRESS", Start: second, End: second + len("alice@x.com"), Confidence: 1.0},
	}

	got, _, err := MaskText(text, spans, nil, NewAllocator(defaultCfg()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "[EMAIL_ADDRESS_0] wrote to [EMAIL_ADDRESS_0]"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// Case: Distinct values.
func TestMaskText_DistinctValues(t *testing.T) {
	text := "alice@x.com and bob@x.com"
	aliceStart, aliceEnd := byteRange(t, text, "alice@x.com")
	bobStart, bobEnd := byteRange(t, text, "bob@x.com")
	spans := []Span{
		{EntityType: "EMAIL_ADDRESS", Start: aliceStart, End: aliceEnd, Confidence: 1.0},
		{EntityType: "EMAIL_ADDRESS", Start: bobStart, End: bobEnd, Confidence: 1.0},
	}

	got, _, err := MaskText(text, spans, nil, NewAllocator(defaultCfg()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got, "[EMAIL_ADDRESS_0]") || !strings.Contains(got, "[EMAIL_ADDRESS_1]") {
		t.Fatalf("expected both _0 and _1 placeholders, got %q", got)
	}
}

// Case: Nested overlap. This is the corruption measured in §0.5: a PERSON
// span over "John Smith" and a narrower FIRST_NAME span over "John" must
// collapse into one clean splice, not two overlapping ones.
func TestMaskText_NestedOverlap(t *testing.T) {
	text := "call John Smith now"
	personStart, personEnd := byteRange(t, text, "John Smith")
	firstNameStart, firstNameEnd := byteRange(t, text, "John")
	spans := []Span{
		{EntityType: "PERSON", Start: personStart, End: personEnd, Confidence: 1.0},
		{EntityType: "FIRST_NAME", Start: firstNameStart, End: firstNameEnd, Confidence: 1.0},
	}

	got, _, err := MaskText(text, spans, nil, NewAllocator(defaultCfg()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "call [PERSON_0] now"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// Case: Order independence. Same two nested spans, reversed input order, must
// produce byte-identical output. This is the test that would catch a
// regression to unstable sorting.
func TestMaskText_OrderIndependence(t *testing.T) {
	text := "call John Smith now"
	personStart, personEnd := byteRange(t, text, "John Smith")
	firstNameStart, firstNameEnd := byteRange(t, text, "John")

	forward := []Span{
		{EntityType: "PERSON", Start: personStart, End: personEnd, Confidence: 1.0},
		{EntityType: "FIRST_NAME", Start: firstNameStart, End: firstNameEnd, Confidence: 1.0},
	}
	reversed := []Span{
		{EntityType: "FIRST_NAME", Start: firstNameStart, End: firstNameEnd, Confidence: 1.0},
		{EntityType: "PERSON", Start: personStart, End: personEnd, Confidence: 1.0},
	}

	gotForward, _, err := MaskText(text, forward, nil, NewAllocator(defaultCfg()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	gotReversed, _, err := MaskText(text, reversed, nil, NewAllocator(defaultCfg()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotForward != gotReversed {
		t.Fatalf("order dependence detected: forward=%q reversed=%q", gotForward, gotReversed)
	}
}

// Case: Adjacent, not overlapping. Two spans that touch at a boundary must
// stay separate, not merge into one.
func TestMaskText_AdjacentSpansNotMerged(t *testing.T) {
	text := "abcdefgh"
	spans := []Span{
		{EntityType: "TYPE_A", Start: 0, End: 4, Confidence: 1.0},
		{EntityType: "TYPE_B", Start: 4, End: 8, Confidence: 1.0},
	}

	got, _, err := MaskText(text, spans, nil, NewAllocator(defaultCfg()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "[TYPE_A_0][TYPE_B_0]"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// Case: Non-ASCII offsets. "José" contains a two-byte rune, so byte offsets
// and rune offsets diverge here. This fails if anyone switches the splice to
// rune-based indexing (D2).
func TestMaskText_NonASCIIByteOffsets(t *testing.T) {
	text := "contact José at j@x.com"
	personStart, personEnd := byteRange(t, text, "José")
	emailStart, emailEnd := byteRange(t, text, "j@x.com")
	spans := []Span{
		{EntityType: "PERSON", Start: personStart, End: personEnd, Confidence: 1.0},
		{EntityType: "EMAIL_ADDRESS", Start: emailStart, End: emailEnd, Confidence: 1.0},
	}

	got, _, err := MaskText(text, spans, nil, NewAllocator(defaultCfg()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "contact [PERSON_0] at [EMAIL_ADDRESS_0]"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// Case: Span past end.
func TestMaskText_SpanPastEndErrors(t *testing.T) {
	text := "short"
	spans := []Span{{EntityType: "PERSON", Start: 0, End: len(text) + 1, Confidence: 1.0}}

	got, citations, err := MaskText(text, spans, nil, NewAllocator(defaultCfg()))
	if err == nil {
		t.Fatalf("expected error, got output %q", got)
	}
	if got != "" || citations != nil {
		t.Fatalf("expected no output on error, got text %q citations %v", got, citations)
	}
}

// Case: Inverted span.
func TestMaskText_InvertedSpanErrors(t *testing.T) {
	text := "short"
	spans := []Span{{EntityType: "PERSON", Start: 3, End: 3, Confidence: 1.0}}

	got, citations, err := MaskText(text, spans, nil, NewAllocator(defaultCfg()))
	if err == nil {
		t.Fatalf("expected error, got output %q", got)
	}
	if got != "" || citations != nil {
		t.Fatalf("expected no output on error, got text %q citations %v", got, citations)
	}
}

// Case: Below threshold.
func TestMaskText_BelowThresholdNotMasked(t *testing.T) {
	text := "call John Smith now"
	start, end := byteRange(t, text, "John Smith")
	spans := []Span{{EntityType: "PERSON", Start: start, End: end, Confidence: 0.5}}
	cfg := &config.MaskingPluginConfig{Threshold: 0.85}

	got, _, err := MaskText(text, spans, nil, NewAllocator(cfg))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != text {
		t.Fatalf("expected unmasked text %q, got %q", text, got)
	}
}

// Case: Type not included.
func TestMaskText_TypeNotIncludedNotMasked(t *testing.T) {
	text := "call John Smith now"
	start, end := byteRange(t, text, "John Smith")
	spans := []Span{{EntityType: "PERSON", Start: start, End: end, Confidence: 1.0}}
	cfg := &config.MaskingPluginConfig{EntityTypes: []string{"EMAIL_ADDRESS"}}

	got, _, err := MaskText(text, spans, nil, NewAllocator(cfg))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != text {
		t.Fatalf("expected unmasked text %q, got %q", text, got)
	}
}

// Case: Empty entity_types masks every type.
func TestMaskText_EmptyEntityTypesMasksAll(t *testing.T) {
	text := "call John Smith now"
	start, end := byteRange(t, text, "John Smith")
	spans := []Span{{EntityType: "PERSON", Start: start, End: end, Confidence: 1.0}}
	cfg := &config.MaskingPluginConfig{EntityTypes: nil}

	got, _, err := MaskText(text, spans, nil, NewAllocator(cfg))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "call [PERSON_0] now"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// Case: Citation before span is unchanged.
func TestMaskText_CitationBeforeSpanUnchanged(t *testing.T) {
	text := "mail alice@x.com"
	start, end := byteRange(t, text, "alice@x.com")
	spans := []Span{{EntityType: "EMAIL_ADDRESS", Start: start, End: end, Confidence: 1.0}}
	citations := []llmprotocol.Citation{{StartIndex: 0, EndIndex: 4}}

	_, got, err := MaskText(text, spans, citations, NewAllocator(defaultCfg()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 1 || got[0].StartIndex != 0 || got[0].EndIndex != 4 {
		t.Fatalf("expected citation unchanged at [0,4), got %v", got)
	}
}

// Case: Citation after span is shifted by the net length delta.
func TestMaskText_CitationAfterSpanShifted(t *testing.T) {
	text := "mail alice@x.com"
	start, end := byteRange(t, text, "alice@x.com") // [5,16), length 11
	spans := []Span{{EntityType: "EMAIL_ADDRESS", Start: start, End: end, Confidence: 1.0}}
	// "XXXXXXX{index}" resolves to "XXXXXXX0" (length 8): the span shortens
	// by exactly 3 bytes, matching the plan's worked example.
	cfg := &config.MaskingPluginConfig{Placeholders: map[string]string{"EMAIL_ADDRESS": "XXXXXXX{index}"}}
	citations := []llmprotocol.Citation{{StartIndex: 17, EndIndex: 20}}

	_, got, err := MaskText(text, spans, citations, NewAllocator(cfg))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 1 || got[0].StartIndex != 14 || got[0].EndIndex != 17 {
		t.Fatalf("expected citation shifted to [14,17), got %v", got)
	}
}

// Case: Citation overlapping a masked span is dropped.
func TestMaskText_CitationOverlappingSpanDropped(t *testing.T) {
	text := "mail alice@x.com"
	start, end := byteRange(t, text, "alice@x.com")
	spans := []Span{{EntityType: "EMAIL_ADDRESS", Start: start, End: end, Confidence: 1.0}}
	citations := []llmprotocol.Citation{{StartIndex: 0, EndIndex: 20}}

	_, got, err := MaskText(text, spans, citations, NewAllocator(defaultCfg()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected overlapping citation to be dropped, got %v", got)
	}
}

// Regression: llmprotocol.Citation offsets are Unicode code points
// (llmprotocol/types.go), not bytes like Span offsets (D2). A leading
// multi-byte rune before the masked span makes the span's byte-end (8)
// diverge from its code-point end (7); a citation starting exactly at the
// code-point end is adjacent, not overlapping, and must be kept and shifted.
// A byte-based comparison would wrongly treat it as overlapping and drop it.
func TestMaskText_CitationOffsetsAreCodePointsNotBytes(t *testing.T) {
	text := "é ALICE end"
	start, end := byteRange(t, text, "ALICE") // byte range [3,8): matches classifier convention
	spans := []Span{{EntityType: "PERSON", Start: start, End: end, Confidence: 1.0}}
	cfg := &config.MaskingPluginConfig{Placeholders: map[string]string{"PERSON": "Y"}}
	// Code-point end of the span is 7 (é=1 code point, space=1, ALICE=5),
	// even though its byte end is 8 (é is 2 bytes). This citation starts
	// exactly there.
	citations := []llmprotocol.Citation{{StartIndex: 7, EndIndex: 9}}

	_, gotCitations, err := MaskText(text, spans, citations, NewAllocator(cfg))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(gotCitations) != 1 {
		t.Fatalf("expected the adjacent citation to survive, got %v", gotCitations)
	}
	if gotCitations[0].StartIndex != 3 || gotCitations[0].EndIndex != 5 {
		t.Fatalf("expected citation shifted to [3,5), got [%d,%d)", gotCitations[0].StartIndex, gotCitations[0].EndIndex)
	}
}

// Case: Allocator across calls — the per-type counter must not reuse an
// index for a different value on a later MaskText call (D5).
func TestAllocator_AcrossCallsNoIndexReuse(t *testing.T) {
	a := NewAllocator(defaultCfg())

	firstText := "alice@x.com"
	firstSpans := []Span{{EntityType: "EMAIL_ADDRESS", Start: 0, End: len(firstText), Confidence: 1.0}}
	firstGot, _, err := MaskText(firstText, firstSpans, nil, a)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "[EMAIL_ADDRESS_0]"; firstGot != want {
		t.Fatalf("got %q, want %q", firstGot, want)
	}

	secondText := "bob@x.com"
	secondSpans := []Span{{EntityType: "EMAIL_ADDRESS", Start: 0, End: len(secondText), Confidence: 1.0}}
	secondGot, _, err := MaskText(secondText, secondSpans, nil, a)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "[EMAIL_ADDRESS_1]"; secondGot != want {
		t.Fatalf("got %q, want %q", secondGot, want)
	}
}
