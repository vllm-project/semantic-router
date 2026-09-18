package operatingpoint

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func ckDefinition() Definition {
	d := exampleDefinition()
	d.Version = 3
	d.Input.ReferenceWindowBatchSize = 1
	d.Input.BatchOrder = "ascending original window start"
	d.Executions = []Execution{{Provider: "ort", Precision: "native", WeightsFile: "model.safetensors", ONNX: &ONNXExecution{
		File: "model.onnx", ExecutionProvider: "ROCMExecutionProvider", MaxExecutionTokens: d.Input.WindowTokens,
		CustomOpsProfile: "ck_flash_attention", ExecutionMode: "dynamic_sequence_b1", RuntimeBuild: "qualified ORT build",
		Artifacts: []ArtifactDigest{{Role: "graph", SHA256: strings.Repeat("e", 64)}, {Role: "custom-ops", SHA256: strings.Repeat("f", 64)}},
	}}}
	return d
}

func TestCKPolicyRequiresAnExplicitVersionedExecution(t *testing.T) {
	raw, err := json.Marshal(ckDefinition())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = Decode(raw, digest(raw)); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(string) string{
		"missing profile": func(s string) string { return strings.Replace(s, `"custom_ops_profile":"ck_flash_attention",`, "", 1) },
		"missing mode":    func(s string) string { return strings.Replace(s, `"execution_mode":"dynamic_sequence_b1",`, "", 1) },
		"missing runtime": func(s string) string { return strings.Replace(s, `,"runtime_build":"qualified ORT build"`, "", 1) },
		"empty runtime":   func(s string) string { return strings.Replace(s, "qualified ORT build", "", 1) },
		"wrong profile":   func(s string) string { return strings.Replace(s, "ck_flash_attention", "arbitrary_library", 1) },
		"wrong mode":      func(s string) string { return strings.Replace(s, "dynamic_sequence_b1", "unbounded", 1) },
		"wrong provider":  func(s string) string { return strings.Replace(s, "ROCMExecutionProvider", "CPUExecutionProvider", 1) },
		"wrong precision": func(s string) string { return strings.Replace(s, `"precision":"native"`, `"precision":"fp16"`, 1) },
		"wrong reference batch": func(s string) string {
			return strings.Replace(s, `"reference_window_batch_size":1`, `"reference_window_batch_size":4`, 1)
		},
		"wrong reference order": func(s string) string {
			return strings.Replace(s, "ascending original window start", "ascending actual window token count, stable original order on ties", 1)
		},
		"missing library identity": func(s string) string { return strings.Replace(s, "custom-ops", "external:another-file", 1) },
		"library path selection": func(s string) string {
			return strings.Replace(s, `"execution_mode":`, `"library":"/tmp/untrusted.so","execution_mode":`, 1)
		},
		"unknown version": func(s string) string { return strings.Replace(s, `"version":3`, `"version":4`, 1) },
	} {
		t.Run(name, func(t *testing.T) {
			data := []byte(change(string(raw)))
			if bytes.Equal(data, raw) {
				t.Fatal("test did not mutate policy")
			}
			if _, err := Decode(data, digest(data)); err == nil {
				t.Fatal("unqualified execution accepted")
			}
		})
	}
	// New Go fields must not silently enlarge the frozen v2 schema, even empty.
	legacy := ckDefinition()
	legacy.Version = 2
	legacy.Input.ReferenceWindowBatchSize = 4
	legacy.Input.BatchOrder = "ascending actual window token count, stable original order on ties"
	graph := legacy.Executions[0].ONNX
	graph.ExecutionProvider = "CPUExecutionProvider"
	graph.CustomOpsProfile, graph.ExecutionMode, graph.RuntimeBuild = "", "", ""
	graph.Artifacts = graph.Artifacts[:1]
	raw, _ = json.Marshal(legacy)
	if _, err := Decode(raw, digest(raw)); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"custom_ops_profile", "execution_mode", "runtime_build", "CUSTOM_OPS_PROFILE", "EXECUTION_MODE", "RUNTIME_BUILD"} {
		data := []byte(strings.Replace(string(raw), `"onnx":{`, `"onnx":{"`+key+`":"",`, 1))
		if _, err := Decode(data, digest(data)); err == nil {
			t.Fatalf("v2 accepted v3 field %s", key)
		}
		if _, err := BindArtifact(context.Background(), data, t.TempDir()); err == nil || !strings.Contains(err.Error(), "explicit version-3") {
			t.Fatalf("v2 packaging discarded v3 field %s: %v", key, err)
		}
		data = bytes.Replace(data, []byte(`"version":2`), []byte(`"version":1`), 1)
		if _, err := BindArtifact(context.Background(), data, t.TempDir()); err == nil || !strings.Contains(err.Error(), "explicit version-3") {
			t.Fatalf("v1 packaging discarded v3 field %s: %v", key, err)
		}
	}
}

func ckArtifactFixture(t *testing.T) (config.ResolvedModelBinding, []byte, string) {
	t.Helper()
	spec := fixtureSpec(t, nil)
	original, err := os.ReadFile(filepath.Join(spec.Deployment.Artifact, "point.json"))
	if err != nil {
		t.Fatal(err)
	}
	var source Definition
	if err = json.Unmarshal(original, &source); err != nil {
		t.Fatal(err)
	}
	d := ckDefinition()
	d.ModelWeightsSHA256, d.ModelConfigSHA256, d.TokenizerSHA256 = source.ModelWeightsSHA256, source.ModelConfigSHA256, source.TokenizerSHA256
	graph := []byte("identity fixture, not an executable qualified graph")
	library := []byte("identity fixture, not an executable qualified library")
	d.Executions[0].ONNX.Artifacts[0].SHA256 = digest(graph)
	d.Executions[0].ONNX.Artifacts[1].SHA256 = digest(library)
	if err = os.WriteFile(filepath.Join(spec.Deployment.Artifact, "model.onnx"), graph, 0o600); err != nil {
		t.Fatal(err)
	}
	libraryPath := filepath.Join(t.TempDir(), "trusted-library.so")
	if err = os.WriteFile(libraryPath, library, 0o600); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(spec.Deployment.Artifact, "point.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	spec.Deployment.Provider, spec.Deployment.Device = "ort", "rocm:0"
	spec.Deployment.CustomOpsProfile = "ck_flash_attention"
	spec.Binding.OperatingPoint.SHA256 = digest(raw)
	return spec, raw, libraryPath
}

func TestCKArtifactsBindTrustedLibraryAndRejectReplacement(t *testing.T) {
	spec, raw, library := ckArtifactFixture(t)
	p, err := Decode(raw, digest(raw))
	if err != nil {
		t.Fatal(err)
	}
	p.execution, err = p.selectExecution("ort", "native", "rocm:0")
	if err != nil {
		t.Fatal(err)
	}
	if err = p.verifyArtifacts(context.Background(), spec.Deployment.Artifact, library); err != nil {
		t.Fatal(err)
	}
	if err = p.validateMetadata(spec.Deployment.Artifact); err != nil {
		t.Fatal(err)
	}
	// A same-named model-local library cannot substitute for the installed one.
	if err = os.WriteFile(filepath.Join(spec.Deployment.Artifact, "custom-ops"), []byte("different"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err = p.verifyArtifacts(context.Background(), spec.Deployment.Artifact, library); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(library, []byte("replaced library"), 0o600); err != nil {
		t.Fatal(err)
	}
	if p.verifyArtifacts(context.Background(), spec.Deployment.Artifact, library) == nil {
		t.Fatal("changed trusted library accepted")
	}
	wantError := map[string]string{
		"missing profile":           "does not declare a custom-ops execution",
		"different document budget": "document budget and reject overflow",
		"truncate":                  "document budget and reject overflow",
		"unqualified provider":      "no qualified operating point execution",
	}
	for name, mutate := range map[string]func(*config.ResolvedModelBinding){
		"missing profile":           func(s *config.ResolvedModelBinding) { s.Deployment.CustomOpsProfile = "" },
		"different document budget": func(s *config.ResolvedModelBinding) { s.Deployment.Input.MaxTokens++ },
		"truncate":                  func(s *config.ResolvedModelBinding) { s.Deployment.Input.Overflow = "truncate" },
		"unqualified provider":      func(s *config.ResolvedModelBinding) { s.Deployment.Device = "migraphx:0" },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := spec
			mutate(&candidate)
			if _, err := Load(context.Background(), candidate, p.Labels()); err == nil || !strings.Contains(err.Error(), wantError[name]) {
				t.Fatalf("expected deployment rejection before library access: %v", err)
			}
		})
	}
}

func TestCKPackagingPreservesFrozenBytesAndIdentities(t *testing.T) {
	spec, source, library := ckArtifactFixture(t)
	// Keep serialized threshold precision; packaging cannot rebind any v3 field.
	source = bytes.Replace(source, []byte("0.2"), []byte("2e-1"), 1)
	got, err := verifyFrozenArtifact(context.Background(), source, spec.Deployment.Artifact, library)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, source) {
		t.Fatal("v3 packaging changed frozen policy bytes")
	}
	path := filepath.Join(spec.Deployment.Artifact, "config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, append(data, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err = BindArtifact(context.Background(), source, spec.Deployment.Artifact); err == nil || !strings.Contains(err.Error(), "config.json SHA256") {
		t.Fatalf("v3 config was rebound: %v", err)
	}
}

func TestCKPolicyRequiresComplete262KDocumentCoverage(t *testing.T) {
	d := ckDefinition()
	d.Input.WindowTokens = 32768
	d.Input.ContentTokens = 32766
	d.Input.Overlap = 16383
	d.Input.Stride = 16383
	d.Input.MaxDocumentTokens = 262144
	d.Executions[0].ONNX.MaxExecutionTokens = 32768
	raw, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	p, err := Decode(raw, digest(raw))
	if err != nil {
		t.Fatal(err)
	}
	result := tasks.WindowedLabelScores{
		ContentTokens: 262142,
		Input:         &tasks.InputUsage{OriginalTokens: 262144, ProcessedTokens: 262144},
	}
	for start := 0; start < result.ContentTokens; start += d.Input.Stride {
		end := min(start+d.Input.ContentTokens, result.ContentTokens)
		result.Windows = append(result.Windows, tasks.LabelScoresWindow{Start: start, End: end, Scores: []float32{.1, .2}})
		if end == result.ContentTokens {
			break
		}
	}
	// Only the final short window carries the positive score. Losing the tail
	// must fail, rather than returning a safe prefix classification.
	tail := len(result.Windows) - 1
	result.Windows[tail].Scores = []float32{.9, .2}
	scores, err := p.Reduce(result)
	if err != nil || scores[0] != .9 {
		t.Fatalf("complete long-document result: %v %v", scores, err)
	}
	for _, window := range result.Windows {
		if window.End-window.Start+2 > p.Window().Size {
			t.Fatal("fixture exceeds physical forward budget")
		}
	}
	missing := result
	missing.Windows = result.Windows[:tail]
	if _, err = p.Reduce(missing); err == nil {
		t.Fatal("missing final window accepted")
	}
	result.ContentTokens++
	result.Input.OriginalTokens++
	result.Input.ProcessedTokens++
	result.Windows[tail].End++
	if _, err = p.Reduce(result); err == nil {
		t.Fatal("document beyond frozen budget accepted")
	}
}
