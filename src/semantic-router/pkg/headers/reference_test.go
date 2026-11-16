package headers

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

func TestReferenceNamesEachHeadersHops(t *testing.T) {
	reference := NewReference([]string{"none"})
	if len(reference.Headers) != len(registry) {
		t.Fatalf("the reference has %d headers, the registry %d", len(reference.Headers), len(registry))
	}
	for i, header := range reference.Headers {
		if header.Name != registry[i].Name || strings.Join(header.Direction, ", ") != registry[i].Direction.String() {
			t.Errorf("reference header %d = %s %v, want %s %s", i, header.Name, header.Direction, registry[i].Name, registry[i].Direction)
		}
	}
	if got := (FromClient | ToProvider).Labels(); !slices.Equal(got, []string{"request", "provider request"}) {
		t.Errorf("Labels() = %v", got)
	}
}

// Trace.Set drops an unregistered key, so a renamed member would vanish
// from the example without this check.
func TestReferenceTraceExampleKeepsEveryMember(t *testing.T) {
	example := exampleTrace()
	if len(example.members) != 5 {
		t.Fatalf("the example trace kept %d of its 5 members: %s", len(example.members), example.Encode())
	}
	if got := NewReference(nil).TraceExample; got != example.Encode() || len(got) > TraceLimitBytes {
		t.Fatalf("TraceExample = %q", got)
	}
}

func TestReferenceJSONKeepsMarkdown(t *testing.T) {
	payload, err := NewReference([]string{"none"}).JSON()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(string(payload), "}\n") || strings.Contains(string(payload), `\u003c`) {
		t.Fatalf("JSON() must end in a newline and keep Markdown unescaped: %.200s", payload)
	}
	var decoded Reference
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.Headers) != len(registry) || len(decoded.WarningCodes) != len(warningCodes) || len(decoded.TraceMembers) != len(traceMembers) {
		t.Fatalf("JSON() lost entries: %d headers, %d warning codes, %d trace members",
			len(decoded.Headers), len(decoded.WarningCodes), len(decoded.TraceMembers))
	}
}
