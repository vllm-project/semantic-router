package classification

import (
	"testing"
	"time"
)

// A token_spans.v1 provider may return nested and overlapping spans. The PII
// adapter must hand both through with byte offsets so masking can take their
// union, and it must close the connector it owns when the classifier is
// retired.
func TestPIIHTTPBackendKeepsOverlappingSpansAndCloses(t *testing.T) {
	text := "Reach me at mailto:jane@example.org for John Smith, 12 Baker Street, London."
	_, cfg := newTokenSpansServer(t, func(inputs string) any {
		if inputs != text {
			t.Fatalf("server got %q", inputs)
		}
		return []map[string]any{
			{"label": "EMAIL_ADDRESS", "score": 0.9, "text": "jane@example.org", "start": 19, "end": 35},
			{"label": "URL", "score": 0.9, "text": "mailto:jane@example.org", "start": 12, "end": 35},
			{"label": "PERSON", "score": 0.9, "text": "John Smith", "start": 40, "end": 50},
			{"label": "ADDRESS", "score": 0.9, "text": "John Smith, 12 Baker Street, London", "start": 40, "end": 75},
		}
	})
	backend, err := newPIIHTTPBackend(cfg, testPIIMapping(), time.Second)
	if err != nil {
		t.Fatalf("newPIIHTTPBackend: %v", err)
	}

	result, err := backend.ClassifyTokens(text)
	if err != nil {
		t.Fatalf("ClassifyTokens: %v", err)
	}
	if len(result.Entities) != 4 {
		t.Fatalf("got %d entities, want all 4 overlapping/nested spans", len(result.Entities))
	}
	for _, e := range result.Entities {
		if got := text[e.Start:e.End]; got != e.Text {
			t.Errorf("%s: byte offsets [%d,%d) slice to %q, want %q", e.EntityType, e.Start, e.End, got, e.Text)
		}
	}

	closer, ok := backend.(interface{ Close() error })
	if !ok {
		t.Fatal("piiHTTPBackend does not expose Close; the classifier close path cannot release its connector")
	}
	if err := closer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}
