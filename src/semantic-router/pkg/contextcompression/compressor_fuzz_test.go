package contextcompression

import (
	"encoding/json"
	"testing"
)

func FuzzCompressToolOutputPreservesJSON(f *testing.F) {
	for _, seed := range []string{
		`{"status":"failed","logs":["first","second"],"code":500}`,
		`[{"引用":"证据一"},{"引用":"证据二"}]`,
		`{"emoji":"🚀","number":42,"enabled":true,"missing":null}`,
		`not-json`,
		`{"malformed":`,
	} {
		f.Add(seed, "find failed evidence")
	}
	f.Fuzz(func(t *testing.T, content string, query string) {
		originalTokens := EstimateTokens(content)
		targetTokens := max(1, originalTokens/2)
		result := CompressToolOutput(
			content,
			query,
			1,
			targetTokens,
		)
		if result.Content == "" && content != "" {
			t.Fatal("compression erased non-empty content")
		}
		if json.Valid([]byte(content)) &&
			result.Applied &&
			!json.Valid([]byte(result.Content)) {
			t.Fatalf("compression corrupted JSON: %q", result.Content)
		}
		if result.Applied && result.CompressedTokens > result.OriginalTokens {
			t.Fatalf(
				"compression increased tokens: before=%d after=%d",
				result.OriginalTokens,
				result.CompressedTokens,
			)
		}
	})
}
