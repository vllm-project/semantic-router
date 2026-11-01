package config

import (
	"math"
	"strings"
	"testing"
)

func TestValidateOptionalNonNegativeFloatFieldsRejectsNonFinite(t *testing.T) {
	NaN, Inf, NegInf := math.NaN(), math.Inf(1), math.Inf(-1)
	fields := []optionalNonNegativeFloatField{
		{"global.router.learning.protection.tuning.switch_margin", &NaN},
		{"global.router.learning.protection.tuning.stability_weight", &Inf},
		{"global.router.learning.adaptation.progress_gate.cooldown_seconds", &NegInf},
	}
	for _, field := range fields {
		err := validateOptionalNonNegativeFloatFields([]optionalNonNegativeFloatField{field})
		if err == nil {
			t.Fatalf("%s: non-finite value accepted", field.name)
		}
		if !strings.Contains(err.Error(), field.name) || !strings.Contains(err.Error(), "must be a finite number") {
			t.Fatalf("%s: error must name the field and require a finite number, got %q", field.name, err)
		}
	}
}

func TestValidateOptionalNonNegativeFloatFieldsAcceptsZeroAndValidValues(t *testing.T) {
	zero, valid := 0.0, 1.5
	err := validateOptionalNonNegativeFloatFields([]optionalNonNegativeFloatField{
		{"global.router.learning.protection.tuning.switch_margin", &zero},
		{"global.router.learning.protection.tuning.stability_weight", &valid},
	})
	if err != nil {
		t.Fatalf("zero and valid values must pass, got %v", err)
	}
}

func TestProtectionTuningRejectsNonFiniteSwitchMargin(t *testing.T) {
	err := validateProtectionTuning(
		"global.router.learning.protection.tuning",
		RouterLearningProtectionTuning{SwitchMargin: func() *float64 { v := math.NaN(); return &v }()},
	)
	if err == nil {
		t.Fatal("non-finite switch_margin must be rejected")
	}
	if !strings.Contains(err.Error(), "switch_margin") || !strings.Contains(err.Error(), "must be a finite number") {
		t.Fatalf("error must name the tunable and require a finite number, got %q", err)
	}
}

func TestParseYAMLBytesRejectsNonFiniteGlobalOverrideEncoding(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: cheap
  decisions:
    - name: route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: cheap
global:
  router:
    learning:
      enabled: true
      protection:
        enabled: true
        scope: session
        identity:
          headers:
            session: x-session-id
            conversation: x-conversation-id
        tuning:
          switch_margin: .nan
          stability_weight: 1.0
      state_store:
        backend: redis
        ttl_seconds: 86400
        timeout_ms: 50
        redis:
          address: redis:6379
          database: 2
          key_prefix: "vsr:router-session:v1:"
`))
	if err == nil {
		t.Fatal("ParseYAMLBytes accepted a non-finite protection tunable")
	}
	if !strings.Contains(err.Error(), "NaN") {
		t.Fatalf("error must surface the non-finite value, got %q", err)
	}
}
