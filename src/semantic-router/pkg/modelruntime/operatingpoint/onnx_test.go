package operatingpoint

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func exampleONNX() Execution {
	return Execution{Provider: "ort", Precision: "native", WeightsFile: "model.safetensors", ONNX: &ONNXExecution{File: "onnx/model.onnx", ExecutionProvider: "CPUExecutionProvider", MaxExecutionTokens: 5, Artifacts: []ArtifactDigest{{Role: "graph", SHA256: strings.Repeat("d", 64)}, {Role: "external:model.onnx.data", SHA256: strings.Repeat("e", 64)}}}}
}

func TestONNXExecutionRequiresCompleteCanonicalIdentity(t *testing.T) {
	d := exampleDefinition()
	d.Executions = append(d.Executions, exampleONNX())
	raw, _ := json.Marshal(d)
	if _, err := Decode(raw, digest(raw)); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*Definition){
		"conversion":           func(d *Definition) { d.Executions[1].Precision = "fp16" },
		"missing graph":        func(d *Definition) { d.Executions[1].ONNX = nil },
		"wrong window":         func(d *Definition) { d.Executions[1].ONNX.MaxExecutionTokens = 10 },
		"unqualified provider": func(d *Definition) { d.Executions[1].ONNX.ExecutionProvider = "ROCMExecutionProvider" },
		"escape":               func(d *Definition) { d.Executions[1].ONNX.File = "../model.onnx" },
		"external escape":      func(d *Definition) { d.Executions[1].ONNX.Artifacts[1].Role = "external:../weights.data" },
		"duplicate artifact":   func(d *Definition) { d.Executions[1].ONNX.Artifacts[1].Role = "graph" },
		"missing graph hash":   func(d *Definition) { d.Executions[1].ONNX.Artifacts = d.Executions[1].ONNX.Artifacts[1:] },
		"duplicate execution":  func(d *Definition) { d.Executions = append(d.Executions, d.Executions[1]) },
		"Candle with graph":    func(d *Definition) { d.Executions[0].ONNX = d.Executions[1].ONNX },
	} {
		t.Run(name, func(t *testing.T) {
			var bad Definition
			if err := json.Unmarshal(raw, &bad); err != nil {
				t.Fatal(err)
			}
			change(&bad)
			data, _ := json.Marshal(bad)
			if _, err := Decode(data, digest(data)); err == nil {
				t.Fatal("invalid graph contract accepted")
			}
		})
	}
	for _, changed := range []string{
		strings.Replace(string(raw), `"max_execution_tokens":5,`, "", 1),
		strings.Replace(string(raw), `"execution_provider":`, `"Execution_Provider":`, 1),
		strings.Replace(string(raw), `"onnx":{`, `"onnx":{"extra":1,`, 1),
	} {
		if _, err := Decode([]byte(changed), digest([]byte(changed))); err == nil {
			t.Fatal("noncanonical graph schema accepted")
		}
	}
}

func TestLoadSelectsOnlyQualifiedExecutionAndBindsExternalData(t *testing.T) {
	ctx := context.Background()
	spec := fixtureSpec(t, nil)
	root := spec.Deployment.Artifact
	if err := os.Mkdir(filepath.Join(root, "onnx"), 0o700); err != nil {
		t.Fatal(err)
	}
	graph, external := []byte("fake graph"), []byte("fake external tensor data")
	for path, data := range map[string][]byte{"onnx/model.onnx": graph, "onnx/model.onnx.data": external} {
		if err := os.WriteFile(filepath.Join(root, path), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	original, err := os.ReadFile(filepath.Join(root, "point.json"))
	if err != nil {
		t.Fatal(err)
	}
	var d Definition
	if err = json.Unmarshal(original, &d); err != nil {
		t.Fatal(err)
	}
	execution := exampleONNX()
	execution.ONNX.Artifacts[0].SHA256, execution.ONNX.Artifacts[1].SHA256 = digest(graph), digest(external)
	d.Executions = append(d.Executions, execution)
	data, _ := json.Marshal(d)
	if err = os.WriteFile(filepath.Join(root, "point.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	spec.Binding.OperatingPoint.SHA256 = digest(data)
	spec.Deployment.Provider = "ort"
	policy, err := Load(ctx, spec, d.Labels)
	if err != nil {
		t.Fatal(err)
	}
	selected := policy.ONNX()
	selected.Artifacts[0].SHA256 = "mutated"
	if policy.ONNX().Artifacts[0].SHA256 != digest(graph) {
		t.Fatal("policy mutated")
	}
	packed, err := BindArtifact(ctx, data, root)
	if err != nil {
		t.Fatal(err)
	}
	var roundtrip Definition
	if err = json.Unmarshal(packed, &roundtrip); err != nil {
		t.Fatal(err)
	}
	if len(roundtrip.Executions) != 2 || roundtrip.Executions[1].ONNX.Artifacts[0].SHA256 != digest(graph) {
		t.Fatal("packaging discarded the qualified execution")
	}
	for _, device := range []string{"migraphx:0", "rocm:0"} {
		bad := spec
		bad.Deployment.Device = device
		if _, err = Load(ctx, bad, d.Labels); err == nil {
			t.Fatal("unqualified EP accepted")
		}
	}
	spec.Binding.Head = "onnx/other.onnx"
	if _, err = Load(ctx, spec, d.Labels); err == nil {
		t.Fatal("different graph selected")
	}
	spec.Binding.Head = execution.ONNX.File
	if _, err = Load(ctx, spec, d.Labels); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(root, "onnx/model.onnx.data"), []byte("replacement"), 0o600); err != nil {
		t.Fatal(err)
	}
	if policy.VerifyArtifacts(ctx, root) == nil {
		t.Fatal("replaced external weights reused")
	}
	if _, err = BindArtifact(ctx, data, root); err == nil {
		t.Fatal("packaging accepted replaced external data")
	}
	// A Candle deployment need not download an unselected graph.
	spec.Deployment.Provider, spec.Binding.Head = "candle", ""
	if _, err = Load(ctx, spec, d.Labels); err != nil {
		t.Fatal("unselected ONNX graph blocked Candle", err)
	}
}
