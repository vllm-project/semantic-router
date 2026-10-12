package gatewayparity

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// domainConfig declares the domains math and other, as the domain page does,
// over a remote domain classifier at CLASSIFIER_ADDRESS:CLASSIFIER_PORT.
const domainConfig = `
version: v0.3
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 30s
providers:
  defaults:
    model: math-model
  models:
    - name: math-model
      provider_model_id: math-model
      api_format: openai
      backend_refs:
        - name: math
          endpoint: MATH
          protocol: http
          provider: vllm
    - name: general-model
      provider_model_id: general-model
      api_format: openai
      backend_refs:
        - name: general
          endpoint: GENERAL
          protocol: http
          provider: vllm
routing:
  modelCards:
    - name: math-model
    - name: general-model
  signals:
    domains:
      - name: math
        description: Mathematics
        mmlu_categories: [math]
      - name: other
        description: General fallback traffic
        mmlu_categories: [other]
  decisions:
    - name: math_route
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: domain
            name: math
      modelRefs:
        - model: math-model
          use_reasoning: false
    - name: other_route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: domain
            name: other
      modelRefs:
        - model: general-model
          use_reasoning: false
global:
  model_catalog:
    external:
      - name: domain-service
        model_role: classification
        llm_endpoint:
          address: CLASSIFIER_ADDRESS
          port: CLASSIFIER_PORT
          protocol: http
        llm_model_name: domain-service
        llm_timeout_seconds: 5
    modules:
      classifier:
        domain:
          category_mapping_path: MAPPING_PATH
          threshold: 0.5
          backend:
            protocol: http_classify
            contract: label_distribution.v1
            model: domain-service
            deadline_ms: 5000
`

var domainLabels = []string{"computer science", "law", "math", "other"}

// labelClassifier answers every request with confidence 0.9 on its label.
func labelClassifier(label string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		scores := make([]map[string]any, 0, len(domainLabels))
		for _, name := range domainLabels {
			score := 0.1 / float64(len(domainLabels)-1)
			if name == label {
				score = 0.9
			}
			scores = append(scores, map[string]any{"label": name, "score": score})
		}
		_ = json.NewEncoder(w).Encode(scores)
	})
}

func domainGateway(t *testing.T, label string, backends map[string]http.Handler, native bool) *httptest.Server {
	t.Helper()
	classifier := httptest.NewServer(labelClassifier(label))
	t.Cleanup(classifier.Close)
	address, err := url.Parse(classifier.URL)
	if err != nil {
		t.Fatal(err)
	}
	toIndex, toLabel := map[string]int{}, map[string]string{}
	for index, name := range domainLabels {
		toIndex[name], toLabel[strconv.Itoa(index)] = index, name
	}
	mappingPath := filepath.Join(t.TempDir(), "category_mapping.json")
	encoded, err := json.Marshal(map[string]any{"category_to_idx": toIndex, "idx_to_category": toLabel})
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(mappingPath, encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	configYAML := strings.NewReplacer(
		"CLASSIFIER_ADDRESS", address.Hostname(),
		"CLASSIFIER_PORT", address.Port(),
		"MAPPING_PATH", mappingPath,
	).Replace(domainConfig)
	return gatewayOver(t, configYAML, backends, native)
}

// A label the configuration doesn't declare counts as other: the request
// takes the decision on other, in both gateway modes, and the undeclared label
// never shows up as a matched domain.
func TestAnUndeclaredDomainLabelTakesTheOtherDecisionInBothModes(t *testing.T) {
	for _, tc := range []struct {
		label, decision, domains string
		mathHits, generalHits    int32
	}{
		{label: "computer science", decision: "other_route", domains: "other", generalHits: 1},
		{label: "law", decision: "other_route", domains: "other", generalHits: 1},
		{label: "math", decision: "math_route", domains: "math", mathHits: 1},
	} {
		for _, native := range []bool{true, false} {
			math := &scriptedBackend{status: http.StatusOK, body: chatCompletion("from the math model")}
			general := &scriptedBackend{status: http.StatusOK, body: chatCompletion("from the general model")}
			backends := map[string]http.Handler{"MATH": math, "GENERAL": general}
			got := postChatWithHeaders(t, domainGateway(t, tc.label, backends, native), map[string]string{"x-vsr-debug": "true"})

			if got.status != http.StatusOK {
				t.Fatalf("label %q, native=%t: %s", tc.label, native, got)
			}
			if decision := got.header["x-vsr-selected-decision"]; decision != tc.decision {
				t.Errorf("label %q, native=%t: decision %q, want %q", tc.label, native, decision, tc.decision)
			}
			if domains := got.header["x-vsr-matched-domains"]; domains != tc.domains {
				t.Errorf("label %q, native=%t: matched domains %q, want %q", tc.label, native, domains, tc.domains)
			}
			if math.hits.Load() != tc.mathHits || general.hits.Load() != tc.generalHits {
				t.Errorf("label %q, native=%t: math hits %d, general hits %d", tc.label, native, math.hits.Load(), general.hits.Load())
			}
		}
	}
}
