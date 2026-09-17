package native

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/operatingpoint"
)

func ortPolicyFixture(t *testing.T) (*operatingpoint.Policy, config.ResolvedModelBinding) {
	t.Helper()
	root := t.TempDir()
	source := "../../../../../onnx-binding/instance/testdata/sequence"
	hashes := map[string]string{}
	write := func(name string, data []byte) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(root, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		hashes[name] = hex.EncodeToString(sum[:])
	}
	for _, name := range []string{"config.json", "tokenizer.json", "model.onnx"} {
		data, err := os.ReadFile(filepath.Join(source, name))
		if err != nil {
			t.Fatal(err)
		}
		if name == "config.json" {
			var metadata map[string]any
			if err = json.Unmarshal(data, &metadata); err != nil {
				t.Fatal(err)
			}
			metadata["problem_type"], metadata["classifier_pooling"] = "multi_label_classification", "cls"
			metadata["label2id"] = map[string]int{"negative": 0, "positive": 1}
			metadata["max_position_embeddings"] = 32768
			data, err = json.Marshal(metadata)
			if err != nil {
				t.Fatal(err)
			}
		}
		write(name, data)
	}
	// The tiny ONNX graph is self-contained; this marker exercises source identity.
	write("model.safetensors", []byte("source checkpoint identity fixture"))
	zero, pad := 0, uint32(0)
	d := operatingpoint.Definition{Version: 2, ModelWeightsSHA256: hashes["model.safetensors"], ModelConfigSHA256: hashes["config.json"], TokenizerSHA256: hashes["tokenizer.json"], ScoreType: "independent_sigmoid", Labels: []string{"negative", "positive"}, Thresholds: []float32{.5, .7}, Comparison: "score >= threshold", Executions: []operatingpoint.Execution{{Provider: "ort", Precision: "native", WeightsFile: "model.safetensors", ONNX: &operatingpoint.ONNXExecution{File: "model.onnx", ExecutionProvider: "CPUExecutionProvider", MaxExecutionTokens: 2048, Artifacts: []operatingpoint.ArtifactDigest{{Role: "graph", SHA256: hashes["model.onnx"]}}}}}, Input: operatingpoint.InputPolicy{Strategy: "overlapping_content_windows", WindowTokens: 2048, ContentTokens: 2048, Stride: 1024, Overlap: 1024, MaxDocumentTokens: 32768, Aggregation: "per-label maximum sigmoid over all covering windows", Positions: "reset for each window", PaddingSide: "right", PadTokenID: &pad, PaddingAttentionMask: &zero, SpecialPrefixIDs: []uint32{}, SpecialSuffixIDs: []uint32{}, ReferenceWindowBatchSize: 4, BatchOrder: "ascending actual window token count, stable original order on ties", Tokenization: "Tokenize once without truncation; slice original content token IDs and restore the tokenizer special-token envelope for each window.", Overflow: "reject"}}
	data, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	write("point.json", data)
	spec := config.ResolvedModelBinding{Recipe: "one", Name: "classifier.risk", Binding: config.ModelBinding{Deployment: "head", Adapter: "modernbert", Contract: config.RemoteClassifierContractLabelScores, OperatingPoint: &config.OperatingPointReference{Path: "point.json", SHA256: hashes["point.json"]}}, Deployment: config.ModelDeployment{Provider: "ort", Artifact: root, Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 32768, Overflow: "reject"}}}
	policy, err := operatingpoint.Load(context.Background(), spec, d.Labels)
	if err != nil {
		t.Fatal(err)
	}
	return policy, spec
}

func TestOperatingPointRejectsMismatchedActualSession(t *testing.T) {
	policy, spec := ortPolicyFixture(t)
	evidence := ort.SessionEvidence{RuntimeBuild: "test-runtime", Provider: "CPUExecutionProvider", Precision: "native", Graph: filepath.Join(spec.Deployment.Artifact, "model.onnx"), ExecutionMaxInputTokens: 2048, Artifacts: []ort.ArtifactDigest{{Role: "graph", SHA256: policy.ONNX().Artifacts[0].SHA256}}}
	info := ort.Info{Task: "label_scores", Sessions: []ort.SessionEvidence{evidence}}
	if err := validateOperatingPointSession(policy, spec, info); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*ort.SessionEvidence){
		"different graph": func(s *ort.SessionEvidence) { s.Graph = filepath.Join(spec.Deployment.Artifact, "config.json") },
		"changed tensor": func(s *ort.SessionEvidence) {
			s.Artifacts = []ort.ArtifactDigest{{Role: "graph", SHA256: strings.Repeat("0", 64)}}
		},
		"extra tensor": func(s *ort.SessionEvidence) {
			s.Artifacts = append(append([]ort.ArtifactDigest{}, s.Artifacts...), ort.ArtifactDigest{Role: "external:unbound", SHA256: strings.Repeat("0", 64)})
		},
		"missing evidence": func(s *ort.SessionEvidence) { s.Artifacts = nil },
		"wrong capacity":   func(s *ort.SessionEvidence) { s.ExecutionMaxInputTokens = 32768 },
		"conversion":       func(s *ort.SessionEvidence) { s.Precision = "fp16" },
		"unqualified EP":   func(s *ort.SessionEvidence) { s.Provider = "MIGraphXExecutionProvider" },
		"fixed CPU": func(s *ort.SessionEvidence) {
			s.ExecutionInputs = []ort.ExecutionInput{{Name: "input_ids", Dtype: "int64", Shape: []int64{1, 2048}}}
		},
	} {
		t.Run(name, func(t *testing.T) {
			bad := evidence
			change(&bad)
			if validateOperatingPointSession(policy, spec, ort.Info{Task: "label_scores", Sessions: []ort.SessionEvidence{bad}}) == nil {
				t.Fatal("unbound actual execution accepted")
			}
		})
	}
}

func TestOwnedORTOperatingPointFullDocumentAndLifecycle(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires actual owned ORT library")
	}
	policy, spec := ortPolicyFixture(t)
	ctx := context.Background()
	runtime := New(nil)
	first, err := runtime.OperatingPoint(ctx, spec, policy.Labels())
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	siblingSpec := spec
	siblingSpec.Recipe = "two"
	sibling, err := runtime.OperatingPoint(ctx, siblingSpec, policy.Labels())
	if err != nil {
		t.Fatal(err)
	}
	defer sibling.Close()
	text := strings.Repeat("hello ", 32768)
	out, err := first.Score(ctx, "one", text)
	if err != nil {
		t.Fatal(err)
	}
	if out.Input.OriginalTokens != 32768 || out.Input.Truncated || len(out.Windows) != 31 || out.Scores[1] < .73 || out.Scores[1] > .74 {
		t.Fatalf("invalid complete scan: %+v", out)
	}
	for i, window := range out.Windows {
		if window != [2]int{i * 1024, i*1024 + 2048} {
			t.Fatalf("window %d has unexpected coverage: %v", i, window)
		}
	}
	if _, err = first.Score(ctx, "one", text+"hello"); !errors.Is(err, binding.ErrInputLimit) {
		t.Fatalf("overflow accepted: %v", err)
	}
	if _, err = first.Score(ctx, "two", "hello"); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("foreign recipe accepted: %v", err)
	}
	if err = first.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = first.Score(ctx, "one", "hello"); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("closed owner accepted: %v", err)
	}
	if _, err = sibling.Score(ctx, "two", "hello"); err != nil {
		t.Fatal(err)
	}
}

func TestOperatingPointMIGraphXRequiresFixedWindowAndNoFallback(t *testing.T) {
	_, spec := ortPolicyFixture(t)
	path := filepath.Join(spec.Deployment.Artifact, "point.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data = []byte(strings.Replace(string(data), "CPUExecutionProvider", "MIGraphXExecutionProvider", 1))
	if err = os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	spec.Binding.OperatingPoint.SHA256 = hex.EncodeToString(sum[:])
	spec.Deployment.Device = "migraphx:0"
	policy, err := operatingpoint.Load(context.Background(), spec, []string{"negative", "positive"})
	if err != nil {
		t.Fatal(err)
	}
	session := ort.SessionEvidence{RuntimeBuild: "test", Graph: filepath.Join(spec.Deployment.Artifact, "model.onnx"), Provider: "MIGraphXExecutionProvider", Precision: "native", CPUFallbackDisabled: true, ExecutionMaxInputTokens: 2048, Artifacts: []ort.ArtifactDigest{{Role: "graph", SHA256: policy.ONNX().Artifacts[0].SHA256}}, ExecutionInputs: []ort.ExecutionInput{{Name: "input_ids", Dtype: "int64", Shape: []int64{1, 2048}}, {Name: "attention_mask", Dtype: "int64", Shape: []int64{1, 2048}}}}
	if err = validateOperatingPointSession(policy, spec, ort.Info{Task: "label_scores", Sessions: []ort.SessionEvidence{session}}); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*ort.SessionEvidence){
		"fallback":       func(s *ort.SessionEvidence) { s.CPUFallbackDisabled = false },
		"missing inputs": func(s *ort.SessionEvidence) { s.ExecutionInputs = nil },
		"32K execution": func(s *ort.SessionEvidence) {
			s.ExecutionInputs = []ort.ExecutionInput{{Name: "input_ids", Dtype: "int64", Shape: []int64{1, 32768}}}
		},
		"different batch": func(s *ort.SessionEvidence) {
			s.ExecutionInputs = []ort.ExecutionInput{{Name: "input_ids", Dtype: "int64", Shape: []int64{4, 2048}}}
		},
	} {
		t.Run(name, func(t *testing.T) {
			bad := session
			change(&bad)
			if validateOperatingPointSession(policy, spec, ort.Info{Task: "label_scores", Sessions: []ort.SessionEvidence{bad}}) == nil {
				t.Fatal("unqualified geometry accepted")
			}
		})
	}
}
