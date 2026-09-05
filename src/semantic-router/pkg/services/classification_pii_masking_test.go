package services

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

func maskedFor(t *testing.T, text string, detections []classification.PIIDetection) string {
	t.Helper()
	placeholders := buildPIIMaskPlaceholders(detections)
	return buildMaskedPIIText(text, detections, placeholders)
}

func assertNoPIISurvives(t *testing.T, masked string, detections []classification.PIIDetection) {
	t.Helper()
	for _, d := range detections {
		if strings.Contains(masked, d.Text) {
			t.Fatalf("%s %q survived masking: %q", d.EntityType, d.Text, masked)
		}
	}
}

// Single, non-overlapping spans mask exactly as they did before the rewrite.
func TestBuildMaskedPIITextSingleSpansUnchanged(t *testing.T) {
	text := "Contact John Smith at john@example.org today."
	detections := []classification.PIIDetection{
		{EntityType: "PERSON", Text: "John Smith", Start: 8, End: 18, Confidence: 0.9},
		{EntityType: "EMAIL_ADDRESS", Text: "john@example.org", Start: 22, End: 38, Confidence: 0.9},
	}
	got := maskedFor(t, text, detections)
	if want := "Contact [PERSON_0] at [EMAIL_ADDRESS_0] today."; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// PERSON strictly inside ADDRESS: the outer span wins, nothing leaks.
func TestBuildMaskedPIITextNestedSpans(t *testing.T) {
	text := "Send it to John Smith, 12 Baker Street, London."
	detections := []classification.PIIDetection{
		{EntityType: "PERSON", Text: "John Smith", Start: 11, End: 21, Confidence: 0.9},
		{EntityType: "ADDRESS", Text: "John Smith, 12 Baker Street, London", Start: 11, End: 46, Confidence: 0.9},
	}
	got := maskedFor(t, text, detections)
	if want := "Send it to [ADDRESS_0]."; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	assertNoPIISurvives(t, got, detections)
}

// EMAIL inside URL, reported with the inner span first: same result either order.
func TestBuildMaskedPIITextOverlappingSpansOrderIndependent(t *testing.T) {
	text := "Reach me at mailto:jane@example.org for details."
	inner := classification.PIIDetection{EntityType: "EMAIL_ADDRESS", Text: "jane@example.org", Start: 19, End: 35, Confidence: 0.9}
	outer := classification.PIIDetection{EntityType: "URL", Text: "mailto:jane@example.org", Start: 12, End: 35, Confidence: 0.9}
	for name, detections := range map[string][]classification.PIIDetection{
		"inner first": {inner, outer},
		"outer first": {outer, inner},
	} {
		got := maskedFor(t, text, detections)
		if want := "Reach me at [URL_0] for details."; got != want {
			t.Fatalf("%s: got %q, want %q", name, got, want)
		}
		assertNoPIISurvives(t, got, detections)
	}
}

// Partially overlapping spans of different types merge into one range.
func TestBuildMaskedPIITextPartialOverlap(t *testing.T) {
	text := "id 4111 1111 1111 1111 9999 end"
	detections := []classification.PIIDetection{
		{EntityType: "CREDIT_CARD", Text: "4111 1111 1111 1111", Start: 3, End: 22, Confidence: 0.9},
		{EntityType: "PHONE_NUMBER", Text: "1111 1111 9999", Start: 13, End: 27, Confidence: 0.6},
	}
	got := maskedFor(t, text, detections)
	if want := "id [CREDIT_CARD_0] end"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	assertNoPIISurvives(t, got, detections)
}

// Adjacent spans stay two placeholders, one at offset 0 and one at the end.
func TestBuildMaskedPIITextAdjacentAtEdges(t *testing.T) {
	text := "John Smithjohn@example.org"
	detections := []classification.PIIDetection{
		{EntityType: "PERSON", Text: "John Smith", Start: 0, End: 10, Confidence: 0.9},
		{EntityType: "EMAIL_ADDRESS", Text: "john@example.org", Start: 10, End: 26, Confidence: 0.9},
	}
	got := maskedFor(t, text, detections)
	if want := "[PERSON_0][EMAIL_ADDRESS_0]"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// Byte offsets into multi-byte text, with an overlapping pair: no panic, no leak.
func TestBuildMaskedPIITextMultiByteOverlap(t *testing.T) {
	text := "こんにちは、私は John Smith です。電話は 415-555-0134"
	detections := []classification.PIIDetection{
		{EntityType: "PERSON", Text: "John Smith", Start: 25, End: 35, Confidence: 0.99},
		{EntityType: "PHONE_NUMBER", Text: "415-555-0134", Start: 55, End: 67, Confidence: 0.99},
		{EntityType: "PHONE_NUMBER", Text: "555-0134", Start: 59, End: 67, Confidence: 0.7},
	}
	got := maskedFor(t, text, detections)
	if want := "こんにちは、私は [PERSON_0] です。電話は [PHONE_NUMBER_0]"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
	assertNoPIISurvives(t, got, detections)
}

// Out-of-range spans are ignored rather than slicing past the string.
func TestBuildMaskedPIITextIgnoresInvalidSpans(t *testing.T) {
	text := "John Smith called."
	detections := []classification.PIIDetection{
		{EntityType: "PERSON", Text: "John Smith", Start: 0, End: 10, Confidence: 0.9},
		{EntityType: "PHONE_NUMBER", Text: "nope", Start: 12, End: 99, Confidence: 0.9},
	}
	got := maskedFor(t, text, detections)
	if want := "[PERSON_0] called."; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// End to end through the response builder: overlapping and nested spans, as a
// token_spans.v1 provider may return them, must leave no PII in MaskedText and
// must keep every entity reported.
func TestBuildPIIResponse_MaskEntitiesOverlappingAndNested(t *testing.T) {
	service := newPIIResponseTestService()

	t.Run("nested_person_inside_address", func(t *testing.T) {
		text := "Send it to John Smith, 12 Baker Street, London."
		detections := []classification.PIIDetection{
			{EntityType: "PERSON", Start: 11, End: 21, Text: "John Smith", Confidence: 0.9},
			{EntityType: "ADDRESS", Start: 11, End: 46, Text: "John Smith, 12 Baker Street, London", Confidence: 0.9},
		}
		resp := service.buildPIIResponse(text, detections, &PIIOptions{MaskEntities: true, ReturnPositions: true})
		if resp.MaskedText != "Send it to [ADDRESS_0]." {
			t.Fatalf("masked text = %q", resp.MaskedText)
		}
		if len(resp.Entities) != 2 {
			t.Fatalf("entities = %d, want both spans reported", len(resp.Entities))
		}
		assertNoPIISurvives(t, resp.MaskedText, detections)
	})

	t.Run("email_inside_url_reported_inner_first", func(t *testing.T) {
		text := "Reach me at mailto:jane@example.org for details."
		detections := []classification.PIIDetection{
			{EntityType: "EMAIL_ADDRESS", Start: 19, End: 35, Text: "jane@example.org", Confidence: 0.9},
			{EntityType: "URL", Start: 12, End: 35, Text: "mailto:jane@example.org", Confidence: 0.9},
		}
		resp := service.buildPIIResponse(text, detections, &PIIOptions{MaskEntities: true})
		if resp.MaskedText != "Reach me at [URL_0] for details." {
			t.Fatalf("masked text = %q", resp.MaskedText)
		}
		assertNoPIISurvives(t, resp.MaskedText, detections)
	})
}
