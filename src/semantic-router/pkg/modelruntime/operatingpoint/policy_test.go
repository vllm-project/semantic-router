package operatingpoint

import (
	"encoding/json"
	"strings"
	"testing"
)

func exampleDefinition() Definition {
	zero := 0
	pad := uint32(0)
	return Definition{Version: 2, ModelWeightsSHA256: strings.Repeat("a", 64), ModelConfigSHA256: strings.Repeat("b", 64), TokenizerSHA256: strings.Repeat("c", 64), ScoreType: "independent_sigmoid", Labels: []string{"one", "two"}, Thresholds: []float32{.2, .7}, Comparison: "score >= threshold", Executions: []Execution{{Provider: "candle", Precision: "float32", WeightsFile: "model.safetensors"}}, Input: InputPolicy{Strategy: "overlapping_content_windows", WindowTokens: 5, ContentTokens: 3, Stride: 2, Overlap: 1, MaxDocumentTokens: 10, Aggregation: "per-label maximum sigmoid over all covering windows", Positions: "reset for each window", PaddingSide: "right", PadTokenID: &pad, PaddingAttentionMask: &zero, SpecialPrefixIDs: []uint32{2}, SpecialSuffixIDs: []uint32{1}, ReferenceWindowBatchSize: 4, BatchOrder: "ascending actual window token count, stable original order on ties", Tokenization: "Tokenize once without truncation; slice original content token IDs and restore the tokenizer special-token envelope for each window.", Overflow: "reject"}}
}

func TestStrictPolicyDecode(t *testing.T) {
	raw, _ := json.Marshal(exampleDefinition())
	for name, change := range map[string]func(string) string{
		"legacy":                func(s string) string { return strings.Replace(s, `"version":2`, `"version":1`, 1) },
		"unknown":               func(s string) string { return strings.Replace(s, `"version":2`, `"version":2,"unused":true`, 1) },
		"duplicate":             func(s string) string { return strings.Replace(s, `"version":2`, `"version":2,"version":2`, 1) },
		"null threshold":        func(s string) string { return strings.Replace(s, `[0.2,0.7]`, `[null,0.7]`, 1) },
		"missing pad":           func(s string) string { return strings.Replace(s, `"pad_token_id":0,`, ``, 1) },
		"missing overlap":       func(s string) string { return strings.Replace(s, `"overlap_content_tokens":1,`, ``, 1) },
		"case folded duplicate": func(s string) string { return strings.Replace(s, `"version":2`, `"version":2,"Version":2`, 1) },
		"wrong geometry": func(s string) string {
			return strings.Replace(s, `"stride_content_tokens":2`, `"stride_content_tokens":3`, 1)
		},
		"duplicate label":   func(s string) string { return strings.Replace(s, `"two"`, `"one"`, 1) },
		"unqualified graph": func(s string) string { return strings.Replace(s, `"provider":"candle"`, `"provider":"ort"`, 1) },
		"wrong comparison": func(s string) string {
			return strings.Replace(s, `score \u003e= threshold`, `score \u003e threshold`, 1)
		},
		"extra document": func(s string) string { return s + `{}` },
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Decode([]byte(change(string(raw))), strings.Repeat("d", 64)); err == nil {
				t.Fatal("ambiguous/unsupported policy accepted")
			}
		})
	}
}
