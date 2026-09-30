package native

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/operatingpoint"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestModelDiagnosticsUsesPreparedOperatingPoint(t *testing.T) {
	zero := 0
	pad := uint32(0)
	definition := operatingpoint.Definition{Version: 2, ModelWeightsSHA256: strings.Repeat("a", 64), ModelConfigSHA256: strings.Repeat("b", 64), TokenizerSHA256: strings.Repeat("c", 64), ScoreType: "independent_sigmoid", Labels: []string{"one", "two"}, Thresholds: []float32{.2, .7}, Comparison: "score >= threshold", Executions: []operatingpoint.Execution{{Provider: "candle", Precision: "float32", WeightsFile: "model.safetensors"}}, Input: operatingpoint.InputPolicy{Strategy: "overlapping_content_windows", WindowTokens: 5, ContentTokens: 3, Stride: 2, Overlap: 1, MaxDocumentTokens: 10, Aggregation: "per-label maximum sigmoid over all covering windows", Positions: "reset for each window", PaddingSide: "right", PadTokenID: &pad, PaddingAttentionMask: &zero, SpecialPrefixIDs: []uint32{2}, SpecialSuffixIDs: []uint32{1}, ReferenceWindowBatchSize: 4, BatchOrder: "ascending actual window token count, stable original order on ties", Tokenization: "Tokenize once without truncation; slice original content token IDs and restore the tokenizer special-token envelope for each window.", Overflow: "reject"}}
	raw, err := json.Marshal(definition)
	if err != nil {
		t.Fatal(err)
	}
	policy, err := operatingpoint.Decode(raw, strings.Repeat("d", 64))
	if err != nil {
		t.Fatal(err)
	}
	runtime := New(nil)
	resource, err := runtime.Pool.Acquire(context.Background(), binding.ResourceIdentity{Artifact: "controlled-weights", Revision: "frozen", Provider: "test", Device: "cpu", Precision: "fp32"}, "", nil, func(context.Context) (io.Closer, error) { return io.NopCloser(strings.NewReader("")), nil })
	if err != nil {
		t.Fatal(err)
	}
	capability := binding.Capability{Contract: "label_scores.v1", Provider: "test", Device: "cpu", Precision: "fp32", Labels: definition.Labels, Window: &binding.WindowCapability{Size: 5, Overlap: 1}}
	handle, err := runtime.scoreWindows.Resolve(binding.Identity{Recipe: "private", Name: "classifier.risk", Deployment: "frozen", Contract: capability.Contract, Adapter: "test"}, capability, resource, func(_ context.Context, _ io.Closer, input tasks.TextWindowsRequest) (tasks.WindowedLabelScores, error) {
		if input.Size != 5 || input.Overlap != 1 {
			t.Errorf("policy geometry altered: %+v", input)
		}
		return tasks.WindowedLabelScores{ContentTokens: 6, Input: &tasks.InputUsage{OriginalTokens: 8, ProcessedTokens: 8}, Windows: []tasks.LabelScoresWindow{{Start: 0, End: 3, Scores: []float32{.2, .8}}, {Start: 2, End: 5, Scores: []float32{.9, .3}}, {Start: 4, End: 6, Scores: []float32{.1, .7}}}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Close()
	runtime.operatingPoints[handle] = &OperatingPointScorer{handle: handle, policy: policy}
	handle.Ready()
	result, err := runtime.DiagnoseScores(context.Background(), "private", "classifier.risk", "document")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Result.Scores, []float32{.9, .8}) || !reflect.DeepEqual(result.Result.Thresholds, []float32{.2, .7}) || result.Result.PolicySHA256 != strings.Repeat("d", 64) || len(result.Result.WindowRanges) != 3 || result.Result.Windows != nil {
		t.Fatalf("diagnostics bypassed operating point: %+v", result)
	}
	if _, err = runtime.DiagnoseScores(context.Background(), "foreign", "classifier.risk", "document"); !errors.Is(err, binding.ErrNotPrepared) {
		t.Fatalf("foreign policy exposed: %v", err)
	}
	_ = handle.Close()
	if _, err = runtime.DiagnoseScores(context.Background(), "private", "classifier.risk", "document"); !errors.Is(err, binding.ErrNotPrepared) {
		t.Fatalf("retired policy callable: %v", err)
	}
}
