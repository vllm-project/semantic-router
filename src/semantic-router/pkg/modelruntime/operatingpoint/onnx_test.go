package operatingpoint

import (
	"encoding/json"
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
