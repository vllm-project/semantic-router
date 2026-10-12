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
  defaults:
    model: qwen
  models:
    - name: qwen
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: localhost:8000
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
        - model: qwen
entrypoints:
  - model_names: [vllm-sr/private]
    recipe: private
recipes:
  - name: private
    routing:
      signals:
        keywords:
          - name: private-signal
            operator: any
            keywords: ["private"]
      decisions:
        - name: private-route
          priority: 10
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: private-signal
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
	if !strings.Contains(dr.DSL, "ENTRYPOINT {") || !strings.Contains(dr.DSL, "RECIPE private") {
		t.Errorf("expected entrypoints and recipes in decompiled DSL, got:\n%s", dr.DSL)
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

const envReferenceConfigPrefix = `version: v0.3
providers:
  models:
    - name: qwen
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: localhost:8000
          protocol: http
routing:
  modelCards:
    - name: qwen
      modality: text
  signals:
    keywords:
      - name: docs
        operator: any
        keywords: ["docs"]
  decisions:
    - name: docs_route
      priority: 1
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: docs
      modelRefs:
        - model: qwen
      plugins:
`

func decompileAndRecompile(t *testing.T, yamlSource string) (string, string) {
	t.Helper()
	var dr DecompileResult
	if err := json.Unmarshal([]byte(decompile(js.Undefined(), []js.Value{js.ValueOf(yamlSource)}).(string)), &dr); err != nil {
		t.Fatalf("failed to unmarshal decompile result: %v", err)
	}
	if dr.Error != "" {
		t.Fatalf("unexpected decompile error: %s", dr.Error)
	}
	var cr CompileResult
	if err := json.Unmarshal([]byte(compile(js.Undefined(), []js.Value{js.ValueOf(dr.DSL)}).(string)), &cr); err != nil {
		t.Fatalf("failed to unmarshal compile result: %v", err)
	}
	if cr.Error != "" {
		t.Fatalf("unexpected compile error: %s", cr.Error)
	}
	return dr.DSL, cr.YAML
}

func TestDecompileKeepsEnvironmentReferences(t *testing.T) {
	t.Setenv("UPSTREAM_TOKEN", "token-from-environment")
	yamlSource := envReferenceConfigPrefix + `        - type: system_prompt
          configuration:
            enabled: true
            system_prompt: "Quote prices in $$USD."
        - type: header_mutation
          configuration:
            add:
              - name: Authorization
                value: "Bearer ${UPSTREAM_TOKEN}"
              - name: X-Tenant
                value: "${TENANT_ID:-default-tenant}"
`

	dslText, yamlText := decompileAndRecompile(t, yamlSource)
	for _, want := range []string{"$$USD", "Bearer ${UPSTREAM_TOKEN}", "${TENANT_ID:-default-tenant}"} {
		if !strings.Contains(dslText, want) {
			t.Errorf("decompiled DSL lost %q:\n%s", want, dslText)
		}
		if !strings.Contains(yamlText, want) {
			t.Errorf("recompiled YAML lost %q:\n%s", want, yamlText)
		}
	}
	if strings.Contains(dslText, "token-from-environment") {
		t.Errorf("decompiled DSL contains an environment value:\n%s", dslText)
	}
}

func TestDecompileAcceptsRequiredEnvironmentReference(t *testing.T) {
	yamlSource := envReferenceConfigPrefix + `        - type: rag
          configuration:
            enabled: true
            backend: openai
            backend_config:
              vector_store_id: vs_docs
              api_key: ${OPENAI_API_KEY}
`

	dslText, yamlText := decompileAndRecompile(t, yamlSource)
	if !strings.Contains(dslText, `api_key: "${OPENAI_API_KEY}"`) {
		t.Errorf("decompiled DSL lost the API key reference:\n%s", dslText)
	}
	if !strings.Contains(yamlText, "api_key: ${OPENAI_API_KEY}") {
		t.Errorf("recompiled YAML lost the API key reference:\n%s", yamlText)
	}
}

// Each typed value is valid only once its reference takes the default, and the
// RAG key is valid only as written.
const typedEnvDefaultsConfig = `version: v0.3
providers:
  models:
    - name: qwen
      reliability:
        base_ejection_time: ${EJECTION_TIME:-30s}
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: ${BACKEND_HOST:-127.0.0.1}:8000
          protocol: http
routing:
  modelCards:
    - name: qwen
      modality: text
  signals:
    context:
      - name: long_context
        min_tokens: ${CTX_MIN:-4k}
  decisions:
    - name: long_route
      priority: 1
      rules:
        operator: AND
        conditions:
          - type: context
            name: long_context
      modelRefs:
        - model: qwen
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: openai
            backend_config:
              vector_store_id: vs_docs
              api_key: ${OPENAI_API_KEY}
`

func TestDecompileAcceptsTypedEnvironmentDefaults(t *testing.T) {
	// The Router resolves these with its own environment, not this one.
	t.Setenv("CTX_MIN", "lots")
	t.Setenv("EJECTION_TIME", "soon")
	t.Setenv("BACKEND_HOST", "not a host")

	dslText, yamlText := decompileAndRecompile(t, typedEnvDefaultsConfig)
	if !strings.Contains(dslText, `min_tokens: "${CTX_MIN:-4k}"`) {
		t.Errorf("decompiled DSL lost the token count reference:\n%s", dslText)
	}
	for _, want := range []string{"min_tokens: ${CTX_MIN:-4k}", "api_key: ${OPENAI_API_KEY}"} {
		if !strings.Contains(yamlText, want) {
			t.Errorf("recompiled YAML lost %q:\n%s", want, yamlText)
		}
	}
}

func TestDecompileRejectsInvalidEnvironmentDefault(t *testing.T) {
	yamlSource := strings.Replace(typedEnvDefaultsConfig, "${CTX_MIN:-4k}", "${CTX_MIN:-lots}", 1)

	var dr DecompileResult
	if err := json.Unmarshal([]byte(decompile(js.Undefined(), []js.Value{js.ValueOf(yamlSource)}).(string)), &dr); err != nil {
		t.Fatalf("failed to unmarshal decompile result: %v", err)
	}
	if !strings.Contains(dr.Error, "invalid token count format: lots") {
		t.Errorf("decompile error = %q, want the invalid default", dr.Error)
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
	// The YAML reader materializes the default reasoning mode. Specify it in
	// the source as well so this assertion compares canonical YAML bytes.
	dslInput := strings.Replace(validDSL, `MODEL "qwen"`, `MODEL "qwen" (reasoning = false)`, 1)

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

func TestMarshalJSON(t *testing.T) {
	result := marshalJSON(map[string]string{"key": "value"})
	if result != `{"key":"value"}` {
		t.Errorf("unexpected JSON: %s", result)
	}
}
