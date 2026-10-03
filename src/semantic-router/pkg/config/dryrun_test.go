package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type evaluateGoldenCase struct {
	name         string
	file         string
	wantValid    bool
	wantCode     string
	wantStage    string
	wantField    string
	wantSeverity string
}

func TestEvaluateGoldenCorpus(t *testing.T) {
	cases := []evaluateGoldenCase{
		{
			name:      "valid",
			file:      "valid.yaml",
			wantValid: true,
		},
		{
			name:         "unknown_field",
			file:         "unknown-field.yaml",
			wantValid:    false,
			wantCode:     DiagnosticUnknownField,
			wantStage:    StageParse,
			wantField:    "not_a_real_field",
			wantSeverity: SeverityError,
		},
		{
			name:      "parse_error",
			file:      "parse-error.yaml",
			wantValid: false,
			wantCode:  DiagnosticYAMLParseError,
			wantStage: StageParse,
		},
		{
			name:      "reference",
			file:      "reference.yaml",
			wantValid: false,
			wantCode:  DiagnosticReferenceError,
			wantStage: StageResolve,
			wantField: "routing.decisions",
		},
		{
			name:      "cycle",
			file:      "cycle.yaml",
			wantValid: false,
			wantCode:  DiagnosticCycleError,
			wantStage: StageResolve,
			wantField: "routing.projections.scores",
		},
		{
			name:      "quorum",
			file:      "quorum.yaml",
			wantValid: false,
			wantCode:  DiagnosticQuorumError,
			wantField: "routing.decisions",
		},
		{
			name:      "budget",
			file:      "budget.yaml",
			wantValid: false,
			wantCode:  DiagnosticBudgetError,
		},
		{
			name:      "fallback",
			file:      "fallback.yaml",
			wantValid: false,
			wantCode:  DiagnosticFallbackError,
			wantField: "routing.decisions",
		},
		{
			name:      "conflict",
			file:      "conflict.yaml",
			wantValid: false,
			wantCode:  DiagnosticConflict,
		},
		{
			name:      "capability",
			file:      "capability.yaml",
			wantValid: false,
			wantCode:  DiagnosticCapabilityError,
			wantField: "routing.decisions",
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			assertEvaluateGoldenCase(t, testCase, Evaluate(readDryRunTestdata(t, testCase.file), EvaluateOptions{}))
		})
	}
}

func assertEvaluateGoldenCase(t *testing.T, testCase evaluateGoldenCase, result EvaluateResult) {
	t.Helper()
	if result.ContractVersion != DiagnosticContractVersion {
		t.Fatalf("contract_version = %q", result.ContractVersion)
	}
	if result.Valid != testCase.wantValid {
		t.Fatalf("valid = %v, want %v, errors=%v warnings=%v", result.Valid, testCase.wantValid, result.Errors, result.Warnings)
	}
	if testCase.wantCode == "" {
		if len(result.Errors) != 0 {
			t.Fatalf("unexpected errors: %+v", result.Errors)
		}
		return
	}
	found := findDiagnostic(append(append([]Diagnostic{}, result.Errors...), result.Warnings...), testCase.wantCode)
	if found == nil {
		t.Fatalf("missing code %s, errors=%+v warnings=%+v", testCase.wantCode, result.Errors, result.Warnings)
	}
	if testCase.wantStage != "" && found.Stage != testCase.wantStage {
		t.Fatalf("stage = %q, want %q", found.Stage, testCase.wantStage)
	}
	if testCase.wantField != "" && !strings.Contains(found.Field, testCase.wantField) {
		t.Fatalf("field = %q, want substring %q", found.Field, testCase.wantField)
	}
	if testCase.wantSeverity != "" && found.Severity != testCase.wantSeverity {
		t.Fatalf("severity = %q, want %q", found.Severity, testCase.wantSeverity)
	}
}

func TestEvaluateRedactsSecretsAndPreservesEnvNames(t *testing.T) {
	const secret = "super-secret-value"
	result := Evaluate(readDryRunTestdata(t, "redaction.yaml"), EvaluateOptions{
		CompareToActive: true,
		ActiveYAML:      readDryRunTestdata(t, "active.yaml"),
	})
	if !result.Valid {
		t.Fatalf("expected valid redaction document, errors=%+v", result.Errors)
	}
	if strings.Contains(result.NormalizedYAML, secret) {
		t.Fatal("normalized YAML leaked a secret")
	}
	if !strings.Contains(result.NormalizedYAML, "MY_API_KEY") {
		t.Fatalf("normalized YAML dropped api_key_env: %s", result.NormalizedYAML)
	}
	if !strings.Contains(result.NormalizedYAML, RedactedConfigValue) {
		t.Fatalf("normalized YAML missing redaction marker: %s", result.NormalizedYAML)
	}
	if result.Diff == nil {
		t.Fatal("expected diff against active snapshot")
	}
	if leaked := secretInDiff(result.Diff, secret); leaked != "" {
		t.Fatalf("diff leaked secret in %s", leaked)
	}
}

func TestRedactSensitiveConfigValueCoversSchemaSecretContract(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"api_key":         "plain-secret",
		"api_key_env":     "OPENAI_API_KEY",
		"auth_token":      "auth-token",
		"Authorization":   "Bearer authorization-token",
		"x-api-key":       "header-token",
		"access_token":    "access-token",
		"client_secret":   "${CLIENT_SECRET:-literal-fallback}",
		"password":        "$DB_PASSWORD",
		"tokens_per_unit": 100,
		"token_filter":    "keep",
		"llama_stack": map[string]interface{}{
			"auth_token": "llama-stack-bearer",
		},
	}

	redacted, ok := RedactSensitiveConfigValue(input).(map[string]interface{})
	if !ok {
		t.Fatal("expected map result")
	}
	for _, key := range []string{"api_key", "auth_token", "Authorization", "x-api-key", "access_token"} {
		if redacted[key] != RedactedConfigValue {
			t.Fatalf("%s = %v, want %q", key, redacted[key], RedactedConfigValue)
		}
	}
	if redacted["api_key_env"] != "OPENAI_API_KEY" {
		t.Fatalf("api_key_env should remain visible, got %v", redacted["api_key_env"])
	}
	if redacted["password"] != "$DB_PASSWORD" {
		t.Fatalf("pure environment password reference should remain visible, got %v", redacted["password"])
	}
	if redacted["client_secret"] != RedactedConfigValue {
		t.Fatalf("environment reference with literal fallback must be redacted, got %v", redacted["client_secret"])
	}
	if redacted["tokens_per_unit"] != 100 {
		t.Fatal("tokens_per_unit should not be redacted")
	}
	if redacted["token_filter"] != "keep" {
		t.Fatal("token_filter should not be redacted")
	}
	stack := redacted["llama_stack"].(map[string]interface{})
	if stack["auth_token"] != RedactedConfigValue {
		t.Fatalf("canonical llama_stack.auth_token = %v, want %q", stack["auth_token"], RedactedConfigValue)
	}
}

func TestEvaluateRedactsAuthTokenFromActiveDiff(t *testing.T) {
	const activeToken = "active-bearer-token"
	const candidateToken = "candidate-bearer-token"
	document := func(token string) []byte {
		return []byte(`
version: v0.3
listeners: []
providers:
  defaults:
    model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: m1
  decisions:
    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
global:
  stores:
    vector_store:
      enabled: true
      backend_type: llama_stack
      llama_stack:
        endpoint: http://llama-stack:8321
        auth_token: ` + token + `
`)
	}
	result := Evaluate(document(candidateToken), EvaluateOptions{
		CompareToActive: true,
		ActiveYAML:      document(activeToken),
	})
	if leaked := secretInDiff(result.Diff, activeToken); leaked != "" {
		t.Fatalf("diff leaked active auth_token in %s", leaked)
	}
	if leaked := secretInDiff(result.Diff, candidateToken); leaked != "" {
		t.Fatalf("diff leaked candidate auth_token in %s", leaked)
	}
	if result.Diff == nil {
		t.Fatal("expected diff against an active snapshot that includes auth_token")
	}
	found := false
	for _, entry := range result.Diff.Changed {
		if strings.Contains(entry.Field, "auth_token") {
			found = true
			if entry.Old != RedactedConfigValue || entry.New != RedactedConfigValue {
				t.Fatalf("auth_token diff = %+v, want redacted old and new", entry)
			}
		}
	}
	if !found {
		t.Fatalf("expected an auth_token change, got %+v", result.Diff)
	}
}

func TestEvaluateDiffIsBoundedAndSchemaAware(t *testing.T) {
	result := Evaluate(readDryRunTestdata(t, "valid.yaml"), EvaluateOptions{
		CompareToActive: true,
		ActiveYAML:      readDryRunTestdata(t, "active.yaml"),
	})
	if result.Diff == nil {
		t.Fatal("expected diff")
	}
	if result.Diff.Truncated {
		t.Fatal("small diff should not be truncated")
	}
	foundDescription := false
	for _, entry := range result.Diff.Changed {
		if strings.Contains(entry.Field, `routing.modelCards["m1"]`) {
			foundDescription = true
		}
		if strings.Contains(fmtAny(entry.Old), "active-secret") || strings.Contains(fmtAny(entry.New), "active-secret") {
			t.Fatalf("diff exposed active secret: %+v", entry)
		}
	}
	if !foundDescription && len(result.Diff.Changed)+len(result.Diff.Added)+len(result.Diff.Removed) == 0 {
		t.Fatalf("expected a named modelCard or secret-field diff, got %+v", result.Diff)
	}
}

func TestEvaluateCompareWithoutActiveAddsWarning(t *testing.T) {
	result := Evaluate(readDryRunTestdata(t, "valid.yaml"), EvaluateOptions{CompareToActive: true})
	if result.Diff != nil {
		t.Fatalf("expected omitted diff, got %+v", result.Diff)
	}
	if findDiagnostic(result.Warnings, DiagnosticNoActiveSnapshot) == nil {
		t.Fatalf("expected no-active-snapshot warning, got %+v", result.Warnings)
	}
}

func TestDiffCanonicalDocumentsTruncatesAtBound(t *testing.T) {
	active := map[string]interface{}{"items": map[string]interface{}{}}
	candidate := map[string]interface{}{"items": map[string]interface{}{}}
	activeItems := active["items"].(map[string]interface{})
	candidateItems := candidate["items"].(map[string]interface{})
	for i := 0; i < DiffEntryLimit+25; i++ {
		key := fmt.Sprintf("k%03d", i)
		activeItems[key] = i
		candidateItems[key] = i + 1
	}
	diff := DiffCanonicalDocuments(active, candidate)
	if !diff.Truncated {
		t.Fatal("expected truncated flag")
	}
	if got := len(diff.Added) + len(diff.Removed) + len(diff.Changed); got > DiffEntryLimit {
		t.Fatalf("diff size %d exceeds bound %d", got, DiffEntryLimit)
	}
}

func TestEvaluateDoesNotMutateCandidateBytes(t *testing.T) {
	candidate := readDryRunTestdata(t, "valid.yaml")
	before := string(candidate)
	_ = Evaluate(candidate, EvaluateOptions{
		CompareToActive: true,
		ActiveYAML:      readDryRunTestdata(t, "active.yaml"),
	})
	if string(candidate) != before {
		t.Fatal("Evaluate mutated the candidate buffer")
	}
}

func readDryRunTestdata(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "dryrun", name))
	if err != nil {
		t.Fatalf("read testdata %s: %v", name, err)
	}
	return data
}

func findDiagnostic(diagnostics []Diagnostic, code string) *Diagnostic {
	for i := range diagnostics {
		if diagnostics[i].Code == code {
			return &diagnostics[i]
		}
	}
	return nil
}

func secretInDiff(diff *ConfigDiff, secret string) string {
	if diff == nil {
		return ""
	}
	for _, group := range [][]DiffEntry{diff.Added, diff.Removed, diff.Changed} {
		for _, entry := range group {
			if strings.Contains(fmtAny(entry.Old), secret) || strings.Contains(fmtAny(entry.New), secret) {
				return entry.Field
			}
		}
	}
	return ""
}

func fmtAny(value any) string {
	if value == nil {
		return ""
	}
	return fmt.Sprintf("%v", value)
}
