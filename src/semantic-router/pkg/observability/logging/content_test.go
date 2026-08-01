package logging

import (
	"strings"
	"testing"
)

// These tests pin the content-logging policy from issue #2464: a descriptor
// must never carry the content it describes, but must stay correlatable.

func TestContentDescriptorOmitsContent(t *testing.T) {
	const canary = "my social security number is 123-45-6789"

	got := ContentDescriptor(canary)

	if strings.Contains(got, "123-45-6789") || strings.Contains(got, "social security") {
		t.Fatalf("descriptor leaked content: %q", got)
	}
	if !strings.Contains(got, "len=40") {
		t.Errorf("descriptor should report byte length %d, got %q", len(canary), got)
	}
	if !strings.Contains(got, "sha256=") {
		t.Errorf("descriptor should report a correlation hash, got %q", got)
	}
}

func TestContentDescriptorIsStableAndDistinguishing(t *testing.T) {
	a := ContentDescriptor("same query")
	b := ContentDescriptor("same query")
	c := ContentDescriptor("other query")

	if a != b {
		t.Errorf("identical content must yield identical descriptors: %q vs %q", a, b)
	}
	if a == c {
		t.Errorf("different content must yield different descriptors, both %q", a)
	}
}

func TestContentDescriptorEmpty(t *testing.T) {
	if got := ContentDescriptor(""); !strings.Contains(got, "empty") {
		t.Errorf("empty content should be reported as empty, got %q", got)
	}
	if got := ContentDescriptorBytes(nil); !strings.Contains(got, "empty") {
		t.Errorf("nil content should be reported as empty, got %q", got)
	}
}

func TestContentDescriptorBytesMatchesStringForm(t *testing.T) {
	const payload = `{"choices":[{"message":{"content":"secret completion"}}]}`

	fromBytes := ContentDescriptorBytes([]byte(payload))
	fromString := ContentDescriptor(payload)

	if fromBytes != fromString {
		t.Errorf("byte and string descriptors must agree: %q vs %q", fromBytes, fromString)
	}
	if strings.Contains(fromBytes, "secret completion") {
		t.Fatalf("descriptor leaked provider content: %q", fromBytes)
	}
}
