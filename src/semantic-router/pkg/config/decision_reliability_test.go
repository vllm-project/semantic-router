package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const decisionReliabilityDoc = `
version: v0.3
providers:
  models:
    - name: model-a
      provider_model_id: model-a
      api_format: openai
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: model-a
  decisions:
    - name: slow_route
      priority: 10
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: model-a
          use_reasoning: false
      reliability:
        total_timeout: 90s
        per_try_timeout: 0s
        idle_timeout: 30s
        first_byte_timeout: 5s
        retry_count: 0
        retry_on: reset,5xx
        retriable_status_codes: [429]
        retry_back_off_base: 50ms
        retry_back_off_max: 1s
        retry_after_max: 10s
`

func TestDecisionReliabilityRoundTripsCanonicalConfig(t *testing.T) {
	zero := 0
	want := &DecisionReliability{
		TotalTimeout: "90s", PerTryTimeout: "0s", IdleTimeout: "30s", FirstByteTimeout: "5s",
		RetryCount: &zero, RetryOn: "reset,5xx", RetriableStatusCodes: []int{429},
		RetryBackOffBase: "50ms", RetryBackOffMax: "1s", RetryAfterMax: "10s",
	}
	parsed, err := ParseYAMLBytes([]byte(decisionReliabilityDoc))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if got := parsed.Decisions[0].Reliability; !reflect.DeepEqual(got, want) {
		t.Fatalf("reliability = %#v, want %#v", got, want)
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(parsed))
	if err != nil {
		t.Fatal(err)
	}
	reparsed, err := ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("re-import: %v", err)
	}
	if got := reparsed.Decisions[0].Reliability; !reflect.DeepEqual(got, want) {
		t.Fatalf("exported reliability = %#v, want %#v; a 0 retry count must survive", got, want)
	}
}

func TestDecisionReliabilityNamesTheFieldsOnlyTheNativeGatewayHonors(t *testing.T) {
	r := &DecisionReliability{TotalTimeout: "10s", RetryOn: "reset", IdleTimeout: "5s", RetryAfterMax: "3s"}
	if got := r.NativeOnlyFields(); !reflect.DeepEqual(got, []string{"idle_timeout", "retry_after_max"}) {
		t.Fatalf("native-only fields = %v", got)
	}
	if got := (*DecisionReliability)(nil).NativeOnlyFields(); got != nil {
		t.Fatalf("nil block = %v, want none", got)
	}
}

func TestGatewayReliabilityKeepsNativeOnlyFieldsToTheNativeGateway(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(decisionReliabilityDoc))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if err = ValidateGatewayCapabilities(parsed, GatewayStandalone); err != nil {
		t.Fatalf("the native gateway honors every field: %v", err)
	}
	err = ValidateGatewayCapabilities(parsed, GatewayExtProc)
	if err == nil || !strings.Contains(err.Error(), "decision 'slow_route': reliability.idle_timeout, reliability.first_byte_timeout") ||
		!strings.Contains(err.Error(), "--gateway standalone") {
		t.Fatalf("error = %v, want the native-only fields and a pointer to --gateway standalone", err)
	}
	violations := CheckGatewayCapabilities(parsed, GatewayExtProc)
	if len(violations) != 1 || violations[0].Path != "routing.decisions[slow_route].reliability" ||
		violations[0].Capability != CapabilityNativeReliability {
		t.Fatalf("violations = %+v, want one at the decision's reliability block", violations)
	}
	parsed.Decisions[0].Reliability = &DecisionReliability{TotalTimeout: "10s", PerTryTimeout: "2s", RetryOn: "reset"}
	if err := ValidateGatewayCapabilities(parsed, GatewayExtProc); err != nil {
		t.Fatalf("Envoy honors these per request: %v", err)
	}
}

func TestHeaderMutationCannotWriteTheReliabilityHeaders(t *testing.T) {
	doc := strings.Replace(decisionReliabilityDoc, "      reliability:\n", `      plugins:
        - type: header_mutation
          configuration:
            add:
              - {name: x-envoy-max-retries, value: "3"}
              - {name: x-team, value: search}
            update:
              - {name: X-Envoy-Retry-On, value: 5xx}
            delete: [x-envoy-upstream-rq-timeout-ms, x-legacy]
      reliability:
`, 1)
	parsed, err := ParseYAMLBytes([]byte(doc))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	got := parsed.Decisions[0].GetHeaderMutationConfig()
	want := &HeaderMutationPluginConfig{Add: []HeaderPair{{Name: "x-team", Value: "search"}}, Delete: []string{"x-legacy"}}
	if got == nil || !reflect.DeepEqual(got.Add, want.Add) || len(got.Update) != 0 || !reflect.DeepEqual(got.Delete, want.Delete) {
		t.Fatalf("header_mutation = %#v, want only the other headers", got)
	}

	only := strings.Replace(decisionReliabilityDoc, "      reliability:\n", `      plugins:
        - type: header_mutation
          configuration:
            add: [{name: x-envoy-retriable-status-codes, value: "429"}]
      reliability:
`, 1)
	if _, err := ParseYAMLBytes([]byte(only)); err != nil {
		t.Fatalf("a plugin left empty must still load: %v", err)
	}
}

func TestDecisionReliabilityRejectsInvalidValues(t *testing.T) {
	six, negative := 6, -1
	for name, reliability := range map[string]DecisionReliability{
		"too many retries":       {RetryCount: &six},
		"negative retries":       {RetryCount: &negative},
		"status out of range":    {RetriableStatusCodes: []int{700}},
		"unparsable total":       {TotalTimeout: "soon"},
		"negative per try":       {PerTryTimeout: "-1s"},
		"max below base":         {RetryBackOffBase: "100ms", RetryBackOffMax: "50ms"},
		"max below default base": {RetryBackOffMax: "10ms"},
		"zero back-off base":     {RetryBackOffBase: "0s"},
		"zero retry after max":   {RetryAfterMax: "0s"},
	} {
		decision := Decision{Name: "d", Reliability: &reliability}
		if err := validateDecisionReliability(decision); err == nil {
			t.Errorf("%s: accepted %#v", name, reliability)
		}
	}
}

func TestDecisionReliabilityErrorsNameTheDecision(t *testing.T) {
	doc := strings.Replace(decisionReliabilityDoc, "total_timeout: 90s", "total_timeout: later", 1)
	_, err := ParseYAMLBytes([]byte(doc))
	if err == nil || !strings.Contains(err.Error(), "decision 'slow_route': reliability.total_timeout") {
		t.Fatalf("error = %v, want one naming the decision's field", err)
	}
}
