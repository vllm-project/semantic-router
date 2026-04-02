//go:build js && wasm

package main

import (
	"encoding/json"
	"strings"
	"syscall/js"
	"testing"
)

const validDSL = `MODEL qwen {
  modality: "text"
  capabilities: ["chat"]
}

SIGNAL keyword intent { operator: "any" keywords: ["hello"] threshold: 0.8 }

ROUTE r1 {
  PRIORITY 1
  WHEN keyword("intent")
  MODEL "qwen"
}`

func TestCompileValidDSL(t *testing.T) {
	input := js.ValueOf(validDSL)

	result := compile(js.Undefined(), []js.Value{input})
	str, ok := result.(string)
	if !ok {
		t.Fatalf("expected string result, got %T", result)
	}

	var cr CompileResult
	if err := json.Unmarshal([]byte(str), &cr); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if cr.YAML == "" {
		t.Error("expected non-empty YAML output")
	}
	if cr.CRD == "" {
		t.Error("expected non-empty CRD output")
	}
	if cr.Error != "" {
		t.Errorf("unexpected error: %s", cr.Error)
	}
}

func TestCompileInvalidDSL(t *testing.T) {
	input := js.ValueOf(`INVALID SYNTAX !!!`)
	result := compile(js.Undefined(), []js.Value{input})
	str := result.(string)

	var cr CompileResult
	if err := json.Unmarshal([]byte(str), &cr); err != nil {
		t.Fatalf("failed to unmarshal result: %v", err)
	}

	if cr.Error == "" {
		t.Error("expected error for invalid DSL")
	}
}

func TestCompileNoArgs(t *testing.T) {
	result := compile(js.Undefined(), []js.Value{})
	str := result.(string)

	var cr CompileResult
	json.Unmarshal([]byte(str), &cr)
	if cr.Error == "" {
		t.Error("expected error when no args provided")
	}
}

func TestValidateClean(t *testing.T) {
	input := js.ValueOf(validDSL)

	result := validate(js.Undefined(), []js.Value{input})
	str := result.(string)

	var vr ValidateResult
	json.Unmarshal([]byte(str), &vr)
	if vr.ErrorCount != 0 {
		t.Errorf("expected 0 errors, got %d", vr.ErrorCount)
	}
}

func TestValidateWithErrors(t *testing.T) {
	input := js.ValueOf(`SIGNAL keyword s1 { keywords: ["test"] threshold: 2.0 }
ROUTE r1 {
  PRIORITY 1
  WHEN keyword("undefined_signal")
  MODEL "m1"
}`)

	result := validate(js.Undefined(), []js.Value{input})
	str := result.(string)

	var vr ValidateResult
	json.Unmarshal([]byte(str), &vr)
	if len(vr.Diagnostics) == 0 {
		t.Error("expected diagnostics for invalid input")
	}
}

func TestValidateSoftmaxProjectionPartitionDoesNotAddBrowserRuntimeWarning(t *testing.T) {
	input := js.ValueOf(`MODEL qwen {
  modality: "text"
  capabilities: ["chat"]
}

SIGNAL embedding alpha {
  threshold: 0.1
  candidates: ["alpha example"]
}

SIGNAL embedding beta {
  threshold: 0.1
  candidates: ["beta example"]
}

PROJECTION partition intent_partition {
  semantics: "softmax_exclusive"
  members: ["alpha", "beta"]
  default: "alpha"
}`)

	result := validate(js.Undefined(), []js.Value{input})
	str := result.(string)

	var vr ValidateResult
	json.Unmarshal([]byte(str), &vr)
	for _, diag := range vr.Diagnostics {
		if strings.Contains(diag.Message, "browser validation") {
			t.Fatalf("did not expect browser runtime warning for softmax_exclusive partition, got: %s", diag.Message)
		}
	}
}

func TestValidateTestBlocksStillAddBrowserRuntimeWarning(t *testing.T) {
	input := js.ValueOf(`MODEL qwen {
  modality: "text"
  capabilities: ["chat"]
}

SIGNAL keyword intent {
  operator: "any"
  keywords: ["hello"]
  threshold: 0.8
}

ROUTE r1 {
  PRIORITY 1
  WHEN keyword("intent")
  MODEL "qwen"
}

TEST greet_route {
  "hello" -> "r1"
}`)

	result := validate(js.Undefined(), []js.Value{input})
	str := result.(string)

	var vr ValidateResult
	json.Unmarshal([]byte(str), &vr)
	for _, diag := range vr.Diagnostics {
		if strings.Contains(diag.Message, "TEST blocks are parsed in browser validation") {
			return
		}
	}
	t.Fatal("expected TEST block browser runtime warning")
}

func TestValidateNoArgs(t *testing.T) {
	result := validate(js.Undefined(), []js.Value{})
	str := result.(string)

	var vr ValidateResult
	json.Unmarshal([]byte(str), &vr)
	if vr.Error == "" {
		t.Error("expected error when no args provided")
	}
}

func TestDecompileValidYAML(t *testing.T) {
	yamlInput := js.ValueOf(`version: v0.3
providers:
  default_model: qwen
  models:
    - name: qwen
      backend_refs:
        - name: primary
          endpoint: http://localhost:8000/v1
          protocol: http
routing:
  modelCards:
    - name: qwen
      modality: text
  signals:
    keywords:
      - name: s1
        operator: any
        keywords: ["hello"]
  decisions:
    - name: r1
      priority: 1
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: s1
      modelRefs:
        - model: qwen`)

	result := decompile(js.Undefined(), []js.Value{yamlInput})
	str := result.(string)

	var dr DecompileResult
	json.Unmarshal([]byte(str), &dr)
	if dr.DSL == "" {
		t.Error("expected non-empty DSL output")
	}
	if dr.Error != "" {
		t.Errorf("unexpected error: %s", dr.Error)
	}
	if !strings.Contains(dr.DSL, "MODEL qwen") {
		t.Errorf("expected routing model catalog in decompiled DSL, got:\n%s", dr.DSL)
	}
}

func TestDecompileInvalidYAML(t *testing.T) {
	input := js.ValueOf(`{{{invalid yaml`)
	result := decompile(js.Undefined(), []js.Value{input})
	str := result.(string)

	var dr DecompileResult
	json.Unmarshal([]byte(str), &dr)
	if dr.Error == "" {
		t.Error("expected error for invalid YAML")
	}
}

func TestDecompileNoArgs(t *testing.T) {
	result := decompile(js.Undefined(), []js.Value{})
	str := result.(string)

	var dr DecompileResult
	json.Unmarshal([]byte(str), &dr)
	if dr.Error == "" {
		t.Error("expected error when no args provided")
	}
}

func TestFormatValidDSL(t *testing.T) {
	input := js.ValueOf(validDSL)

	result := format(js.Undefined(), []js.Value{input})
	str := result.(string)

	var fr FormatResult
	json.Unmarshal([]byte(str), &fr)
	if fr.DSL == "" {
		t.Error("expected non-empty formatted DSL")
	}
	if fr.Error != "" {
		t.Errorf("unexpected error: %s", fr.Error)
	}
}

func TestFormatInvalidDSL(t *testing.T) {
	input := js.ValueOf(`INVALID !!!`)
	result := format(js.Undefined(), []js.Value{input})
	str := result.(string)

	var fr FormatResult
	json.Unmarshal([]byte(str), &fr)
	if fr.Error == "" {
		t.Error("expected error for invalid DSL")
	}
}

func TestFormatNoArgs(t *testing.T) {
	result := format(js.Undefined(), []js.Value{})
	str := result.(string)

	var fr FormatResult
	json.Unmarshal([]byte(str), &fr)
	if fr.Error == "" {
		t.Error("expected error when no args provided")
	}
}

func TestRoundTrip(t *testing.T) {
	dslInput := validDSL

	// Compile DSL → YAML
	compileResult := compile(js.Undefined(), []js.Value{js.ValueOf(dslInput)})
	var cr CompileResult
	json.Unmarshal([]byte(compileResult.(string)), &cr)
	if cr.Error != "" {
		t.Fatalf("compile error: %s", cr.Error)
	}

	// Decompile YAML → DSL
	decompileResult := decompile(js.Undefined(), []js.Value{js.ValueOf(cr.YAML)})
	var dr DecompileResult
	json.Unmarshal([]byte(decompileResult.(string)), &dr)
	if dr.Error != "" {
		t.Fatalf("decompile error: %s", dr.Error)
	}
	if dr.DSL == "" {
		t.Fatal("decompile produced empty DSL")
	}

	// Re-compile the decompiled DSL → should produce same YAML
	recompileResult := compile(js.Undefined(), []js.Value{js.ValueOf(dr.DSL)})
	var cr2 CompileResult
	json.Unmarshal([]byte(recompileResult.(string)), &cr2)
	if cr2.Error != "" {
		t.Fatalf("re-compile error: %s", cr2.Error)
	}
	if cr2.YAML != cr.YAML {
		t.Errorf("round-trip YAML mismatch:\noriginal:\n%s\nre-compiled:\n%s", cr.YAML, cr2.YAML)
	}
}

func TestConvertDiagnostics(t *testing.T) {
	result := convertDiagnostics(nil)
	if len(result) != 0 {
		t.Errorf("expected 0 diagnostics, got %d", len(result))
	}
}

func TestMarshalJSON(t *testing.T) {
	result := marshalJSON(map[string]string{"key": "value"})
	if result != `{"key":"value"}` {
		t.Errorf("unexpected JSON: %s", result)
	}
}

func TestJoinErrors(t *testing.T) {
	result := joinErrors(nil)
	if result != "[]" {
		t.Errorf("expected empty array, got: %s", result)
	}
}
