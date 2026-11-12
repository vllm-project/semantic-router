package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const decisionReliabilityDoc = `
version: v0.5
providers:
  models:
    - name: model-a
      provider_model_id: model-a
      api_format: openai
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
recipes:
  - name: default
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
	if len(violations) != 1 || violations[0].Path != "recipes[default].decisions[slow_route].reliability" ||
		violations[0].Capability != CapabilityNativeReliability {
		t.Fatalf("violations = %+v, want one at the decision's reliability block", violations)
	}
	parsed.Decisions[0].Reliability = &DecisionReliability{TotalTimeout: "10s", PerTryTimeout: "2s", RetryOn: "reset"}
	if err := ValidateGatewayCapabilities(parsed, GatewayExtProc); err != nil {
		t.Fatalf("Envoy honors these per request: %v", err)
	}
}

// header_mutation cannot write a header the Router owns: the reliability
// headers carry the decision's reliability block, and the others belong to
// the Router or Envoy. Each one fails the load with its own line.
func TestHeaderMutationCannotWriteTheHeadersTheRouterOwns(t *testing.T) {
	doc := strings.Replace(decisionReliabilityDoc, "\n        reliability:\n", `
        plugins:
          - type: header_mutation
            configuration:
              add:
                - {name: x-envoy-max-retries, value: "3"}
                - {name: x-team, value: search}
              update:
                - {name: X-Envoy-Retry-On, value: 5xx}
                - {name: Authorization, value: Bearer forged}
              delete: [x-envoy-upstream-rq-timeout-ms, x-legacy, x-vsr-selected-model]
        reliability:
`, 1)
	_, err := ParseYAMLBytes([]byte(doc))
	if err == nil {
		t.Fatal("header_mutation wrote headers the Router owns, and the configuration loaded")
	}
	const at = "recipes[default].decisions[slow_route].plugins[0].configuration."
	for _, want := range []string{
		at + "add[x-envoy-max-retries]: reserved_header: x-envoy-max-retries carries the decision's retries and timeouts",
		at + "update[X-Envoy-Retry-On]: reserved_header: X-Envoy-Retry-On carries the decision's retries and timeouts",
		at + "update[Authorization]: reserved_header: Authorization is owned by the Router or Envoy",
		at + "delete[0]: reserved_header: x-envoy-upstream-rq-timeout-ms carries the decision's retries and timeouts",
		at + "delete[2]: reserved_header: x-vsr-selected-model is owned by the Router or Envoy",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the load error does not say\n%s\ngot\n%v", want, err)
		}
	}
	if strings.Contains(err.Error(), "x-team") || strings.Contains(err.Error(), "x-legacy") {
		t.Errorf("a header the Router does not own was refused: %v", err)
	}

	own := strings.Replace(decisionReliabilityDoc, "\n        reliability:\n", `
        plugins:
          - type: header_mutation
            configuration:
              add: [{name: x-team, value: search}]
              delete: [x-legacy]
        reliability:
`, 1)
	parsed, err := ParseYAMLBytes([]byte(own))
	if err != nil {
		t.Fatalf("headers the Router does not own must load: %v", err)
	}
	got := parsed.Decisions[0].GetHeaderMutationConfig()
	if got == nil || !reflect.DeepEqual(got.Add, []HeaderPair{{Name: "x-team", Value: "search"}}) || !reflect.DeepEqual(got.Delete, []string{"x-legacy"}) {
		t.Fatalf("header_mutation = %#v, want it as written", got)
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

// A retry_on name Envoy does not know used to be dropped without a word, so a
// typo such as retriable_status_codes silently turned off the retries it
// named. It fails the load in either reliability block, while every name
// Envoy documents loads, such as the gRPC ones in Istio's default policy.
func TestReliabilityRejectsUnknownRetryOnConditions(t *testing.T) {
	providerTypo := strings.Replace(decisionReliabilityDoc, "      api_format: openai\n",
		"      api_format: openai\n      reliability:\n        retry_count: 2\n        retry_on: 5xx,retriable_status_codes\n", 1)
	decisionTypo := strings.Replace(decisionReliabilityDoc, "retry_on: reset,5xx", "retry_on: reset, 5XX", 1)
	for name, tc := range map[string]struct{ doc, want string }{
		"provider": {providerTypo, `providers.models[model-a].reliability.retry_on has unknown condition "retriable_status_codes"`},
		"decision": {decisionTypo, `decision 'slow_route': reliability.retry_on has unknown condition "5XX"`},
	} {
		_, err := ParseYAMLBytes([]byte(tc.doc))
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error = %v, want one containing %q", name, err, tc.want)
		}
	}
	grpc := strings.Replace(decisionReliabilityDoc, "retry_on: reset,5xx", "retry_on: connect-failure,refused-stream,unavailable,cancelled", 1)
	if _, err := ParseYAMLBytes([]byte(grpc)); err != nil {
		t.Fatalf("Envoy's gRPC names must load: %v", err)
	}
}

// A request that matches no decision runs the recipe's plugins layered over
// global.plugins, and a request for a concrete model runs global.plugins, so
// a header the Router owns fails the load at the layer that writes it, even
// when every decision overrides the list that holds it.
func TestHeaderMutationDefaultsCannotWriteTheHeadersTheRouterOwns(t *testing.T) {
	doc := strings.Replace(decisionReliabilityDoc, "\n        reliability:\n", `
        plugins:
          - type: header_mutation
            configuration:
              add: [{name: x-team, value: search}]
              delete: [x-legacy]
        reliability:
`, 1)
	doc = strings.Replace(doc, "recipes:\n  - name: default\n", `recipes:
  - name: default
    plugins:
      - type: header_mutation
        configuration:
          delete: [x-vsr-selected-model]
`, 1) + `global:
  plugins:
    - type: header_mutation
      configuration:
        add: [{name: Authorization, value: Bearer forged}]
`
	_, err := ParseYAMLBytes([]byte(doc))
	if err == nil {
		t.Fatal("the default header_mutation plugins wrote headers the Router owns, and the configuration loaded")
	}
	for _, want := range []string{
		"global.plugins[header_mutation].configuration.add[Authorization]: reserved_header: Authorization is owned by the Router or Envoy",
		"recipes[default].plugins[header_mutation].configuration.delete[0]: reserved_header: x-vsr-selected-model is owned by the Router or Envoy",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the load error does not say\n%s\ngot\n%v", want, err)
		}
	}
	if strings.Contains(err.Error(), "decisions[slow_route]") {
		t.Errorf("a header the decision does not write was refused at the decision: %v", err)
	}
}

// A request that matches no decision runs the default plugins, so automatic
// output beside a router_replay shadow there fails the load, even when every
// decision sets its own output default.
func TestDefaultPluginsCannotPairAutomaticOutputWithAShadow(t *testing.T) {
	doc := strings.Replace(decisionReliabilityDoc, "\n        reliability:\n", `
        plugins:
          - type: request_params
            configuration:
              default_max_tokens: 256
        reliability:
`, 1) + `global:
  plugins:
    - type: request_params
      configuration:
        default_max_tokens: auto
    - type: router_replay
      configuration:
        enabled: true
        shadow: {model: model-a}
`
	_, err := ParseYAMLBytes([]byte(doc))
	if err == nil || !strings.Contains(err.Error(), `"recipes[default].plugins": automatic output does not support a router_replay shadow`) {
		t.Fatalf("load error = %v, want the default plugins' automatic output refused beside a shadow", err)
	}
}
