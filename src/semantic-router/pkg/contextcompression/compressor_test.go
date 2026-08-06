package contextcompression

import (
	"strings"
	"testing"
)

func TestCompressToolOutputKeepsRelevantAndBoundaryChunks(t *testing.T) {
	content := strings.Join([]string{
		"header metadata and source identifier",
		strings.Repeat("unrelated inventory values ", 120),
		"authentication failure in token validator",
		strings.Repeat("unrelated billing records ", 120),
		"footer checksum and provenance",
	}, "\n")

	result := CompressToolOutput(content, "fix authentication token validator", 100, 55)

	if !result.Applied {
		t.Fatal("expected compression to apply")
	}
	if result.CompressedTokens >= result.OriginalTokens {
		t.Fatalf("compression did not reduce tokens: %#v", result)
	}
	for _, expected := range []string{
		"header metadata",
		"authentication failure",
		"footer checksum",
	} {
		if !strings.Contains(result.Content, expected) {
			t.Fatalf("compressed content omitted %q: %s", expected, result.Content)
		}
	}
}

func TestCompressToolOutputPassesThroughBelowThreshold(t *testing.T) {
	content := "small tool output"
	result := CompressToolOutput(content, "small", 100, 50)
	if result.Applied {
		t.Fatal("small content must pass through")
	}
	if result.Content != content {
		t.Fatalf("content changed: %q", result.Content)
	}
}
