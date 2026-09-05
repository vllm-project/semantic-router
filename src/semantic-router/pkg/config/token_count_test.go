package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// tokenCountCase is one entry of testdata/token_count_cases.json, the parsing
// contract shared with the CLI test suite (src/vllm-sr/tests). Exactly one of
// Value or Error is set.
type tokenCountCase struct {
	Input string `json:"input"`
	Value *int64 `json:"value"`
	Error string `json:"error"`
	Note  string `json:"note"`
}

func loadTokenCountCases(t *testing.T) []tokenCountCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "token_count_cases.json"))
	if err != nil {
		t.Fatalf("read token count cases: %v", err)
	}
	var fixture struct {
		Cases []tokenCountCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatalf("decode token count cases: %v", err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("token count fixture has no cases")
	}
	return fixture.Cases
}

// TestTokenCountValueMatchesSharedContract pins TokenCount.Value to the
// fixture the CLI's parse_token_count is also tested against, so the two
// parsers cannot drift apart silently.
func TestTokenCountValueMatchesSharedContract(t *testing.T) {
	for _, tc := range loadTokenCountCases(t) {
		t.Run(tc.Input, func(t *testing.T) {
			if (tc.Value == nil) == (tc.Error == "") {
				t.Fatalf("case %q must set exactly one of value or error", tc.Input)
			}
			got, err := TokenCount(tc.Input).Value()
			if tc.Error != "" {
				if err == nil {
					t.Fatalf("Value(%q) = %d, want error %q", tc.Input, got, tc.Error)
				}
				if !strings.HasPrefix(err.Error(), tc.Error+":") {
					t.Fatalf("Value(%q) error = %q, want prefix %q", tc.Input, err.Error(), tc.Error)
				}
				return
			}
			if err != nil {
				t.Fatalf("Value(%q) returned error %v, want %d", tc.Input, err, *tc.Value)
			}
			if int64(got) != *tc.Value {
				t.Fatalf("Value(%q) = %d, want %d", tc.Input, got, *tc.Value)
			}
		})
	}
}
