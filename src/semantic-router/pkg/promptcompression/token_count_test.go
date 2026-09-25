package promptcompression

import (
	"os"
	"path/filepath"
	"testing"
)

// Counts measured on each fixture with the mmBERT classifier tokenizer
// (jhu-clsp/mmBERT-base, without special tokens) and tiktoken's cl100k_base.
var tokenCountFixtures = []struct {
	file   string
	mmBERT int
	cl100k int
}{
	{"tool_catalog.json", 612, 588},
	{"handler.go.txt", 514, 371},
	{"english.txt", 244, 242},
	{"chinese.txt", 160, 256},
	{"chinese_with_code.txt", 152, 161},
}

func readTokenCountFixture(t *testing.T, file string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "token_count", file))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return string(data)
}

// An undercount skips compression or overruns the budget, so the estimate may
// fall at most a quarter below the smaller measured count. An overcount only
// compresses sooner, so it may reach twice the larger one.
func TestCountTokensApproxStaysNearTokenizerCounts(t *testing.T) {
	for _, tt := range tokenCountFixtures {
		t.Run(tt.file, func(t *testing.T) {
			lo := min(tt.mmBERT, tt.cl100k) * 3 / 4
			hi := max(tt.mmBERT, tt.cl100k) * 2
			if got := CountTokensApprox(readTokenCountFixture(t, tt.file)); got < lo || got > hi {
				t.Errorf("CountTokensApprox = %d, want %d..%d (mmBERT %d, cl100k_base %d)",
					got, lo, hi, tt.mmBERT, tt.cl100k)
			}
		})
	}
}

func TestCompressAppliesBudgetToJSONAndCode(t *testing.T) {
	const budget = 256
	tests := []struct {
		file     string
		prefix   string
		suffix   string
		measured int // smaller tokenizer count for the fixture alone
	}{
		{"tool_catalog.json", "Choose the tool that answers the question at the end. ", " Which tool reads a file from disk?", 588},
		{"handler.go.txt", "Explain why this handler can time out. ", " What should change?", 371},
	}
	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			prompt := tt.prefix + readTokenCountFixture(t, tt.file) + tt.suffix
			result := Compress(prompt, DefaultConfig(budget))
			if result.Ratio >= 1 {
				t.Fatalf("Compress kept the prompt at an estimated %d tokens; the fixture alone measures %d, over the %d-token budget",
					result.OriginalTokens, tt.measured, budget)
			}
			if result.CompressedTokens > budget {
				t.Errorf("CompressedTokens = %d, want at most %d", result.CompressedTokens, budget)
			}
		})
	}
}
