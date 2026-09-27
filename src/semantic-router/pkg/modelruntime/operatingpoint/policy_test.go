package operatingpoint

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func exampleDefinition() Definition {
	zero := 0
	pad := uint32(0)
	return Definition{Version: 2, ModelWeightsSHA256: strings.Repeat("a", 64), ModelConfigSHA256: strings.Repeat("b", 64), TokenizerSHA256: strings.Repeat("c", 64), ScoreType: "independent_sigmoid", Labels: []string{"one", "two"}, Thresholds: []float32{.2, .7}, Comparison: "score >= threshold", Executions: []Execution{{Provider: "candle", Precision: "float32", WeightsFile: "model.safetensors"}}, Input: InputPolicy{Strategy: "overlapping_content_windows", WindowTokens: 5, ContentTokens: 3, Stride: 2, Overlap: 1, MaxDocumentTokens: 10, Aggregation: "per-label maximum sigmoid over all covering windows", Positions: "reset for each window", PaddingSide: "right", PadTokenID: &pad, PaddingAttentionMask: &zero, SpecialPrefixIDs: []uint32{2}, SpecialSuffixIDs: []uint32{1}, ReferenceWindowBatchSize: 4, BatchOrder: "ascending actual window token count, stable original order on ties", Tokenization: "Tokenize once without truncation; slice original content token IDs and restore the tokenizer special-token envelope for each window.", Overflow: "reject"}}
}

func decodeExample(t *testing.T) *Policy {
	t.Helper()
	data, err := json.Marshal(exampleDefinition())
	if err != nil {
		t.Fatal(err)
	}
	p, err := Decode(data, strings.Repeat("d", 64))
	if err != nil {
		t.Fatal(err)
	}
	return p
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
	p := decodeExample(t)
	labels := p.Labels()
	labels[0] = "mutated"
	thresholds := p.Thresholds()
	thresholds[0] = 1
	if p.Labels()[0] != "one" || p.Thresholds()[0] != .2 {
		t.Fatal("policy mutated through accessor")
	}
}

func coveringScores() tasks.WindowedLabelScores {
	return tasks.WindowedLabelScores{ContentTokens: 6, Input: &tasks.InputUsage{OriginalTokens: 8, ProcessedTokens: 8}, Windows: []tasks.LabelScoresWindow{{Start: 0, End: 3, Scores: []float32{.2, .8}}, {Start: 2, End: 5, Scores: []float32{.9, .3}}, {Start: 4, End: 6, Scores: []float32{.1, .7}}}}
}

func TestFrozenWindowReductionAndIncompleteScans(t *testing.T) {
	p := decodeExample(t)
	got, err := p.Reduce(coveringScores())
	if err != nil || !reflect.DeepEqual(got, []float32{.9, .8}) {
		t.Fatalf("max independent scores: %v %v", got, err)
	}
	for name, change := range map[string]func(*tasks.WindowedLabelScores){
		"missing tail":   func(r *tasks.WindowedLabelScores) { r.Windows = r.Windows[:2] },
		"shifted":        func(r *tasks.WindowedLabelScores) { r.Windows[1].Start = 1 },
		"short interior": func(r *tasks.WindowedLabelScores) { r.Windows[0].End = 2 },
		"nonfinite":      func(r *tasks.WindowedLabelScores) { r.Windows[1].Scores[1] = float32(math.NaN()) },
		"label subset":   func(r *tasks.WindowedLabelScores) { r.Windows[1].Scores = r.Windows[1].Scores[:1] },
		"truncated":      func(r *tasks.WindowedLabelScores) { r.Input.Truncated = true },
		"missing usage":  func(r *tasks.WindowedLabelScores) { r.Input = nil },
		"wrong envelope": func(r *tasks.WindowedLabelScores) { r.Input.OriginalTokens = 9; r.Input.ProcessedTokens = 9 },
		"duplicate tail": func(r *tasks.WindowedLabelScores) { r.Windows = append(r.Windows, r.Windows[2]) },
	} {
		t.Run(name, func(t *testing.T) {
			r := coveringScores()
			change(&r)
			if _, err := p.Reduce(r); err == nil {
				t.Fatal("incomplete/malformed scan accepted")
			}
		})
	}
	empty := tasks.WindowedLabelScores{Input: &tasks.InputUsage{OriginalTokens: 2, ProcessedTokens: 2}, Windows: []tasks.LabelScoresWindow{{Scores: []float32{.1, .2}}}}
	if _, err := p.Reduce(empty); err != nil {
		t.Fatal(err)
	}
	over := tasks.WindowedLabelScores{ContentTokens: 9, Input: &tasks.InputUsage{OriginalTokens: 11, ProcessedTokens: 11}, Windows: []tasks.LabelScoresWindow{{Start: 0, End: 3, Scores: []float32{0, 0}}, {Start: 2, End: 5, Scores: []float32{0, 0}}, {Start: 4, End: 7, Scores: []float32{0, 0}}, {Start: 6, End: 9, Scores: []float32{0, 0}}}}
	if _, err := p.Reduce(over); err == nil {
		t.Fatal("document budget ignored")
	}
}

func TestPolicyChecksActualOwnedCapability(t *testing.T) {
	p := decodeExample(t)
	c := binding.Capability{Contract: "label_scores.v1", Provider: "candle", Precision: "float32", Labels: p.Labels(), Limits: binding.Limits{ModelTokens: 32, TaskTokens: 32, DeploymentTokens: 10, Overflow: "window"}}
	if err := p.ValidateCapability(c); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*binding.Capability){func(c *binding.Capability) { c.Precision = "float16" }, func(c *binding.Capability) { c.Contract = "label_distribution.v1" }, func(c *binding.Capability) { c.Labels = []string{"two", "one"} }, func(c *binding.Capability) { c.Limits.TaskTokens = 5 }, func(c *binding.Capability) { c.Provider = "ort" }} {
		bad := c
		mutate(&bad)
		if p.ValidateCapability(bad) == nil {
			t.Fatal("actual capability mismatch accepted")
		}
	}
}
