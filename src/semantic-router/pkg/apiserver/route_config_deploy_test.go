//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHandleConfigPatchMergesExistingConfig(t *testing.T) {
	configPath := writeDeployTestBaseConfig(t)
	doc, rawYAML := executeRouterConfigUpdateAndRead(t, http.MethodPatch, configPath, "new_route")
	routing := requireRoutingSection(t, doc)
	requireRoutingProjectionBlock(t, routing, rawYAML)
	requireFirstDecisionName(t, routing, "new_route")
}

func TestHandleConfigPutReplacesExistingConfig(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")

	baseCfg := minimalDeployTestConfig("old_route")
	baseCfg.Projections.Partitions = []config.ProjectionPartition{{
		Name:        "legacy_partition",
		Semantics:   "softmax_exclusive",
		Temperature: 0.1,
		Members:     []string{"math"},
		Default:     "math",
	}}
	baseYAML := mustMarshalCanonicalConfigYAML(t, baseCfg)
	if err := os.WriteFile(configPath, baseYAML, 0o644); err != nil {
		t.Fatalf("write base config: %v", err)
	}

	doc, deployedYAML := executeRouterConfigUpdateAndRead(t, http.MethodPut, configPath, "new_route")
	if _, hasRootDecisions := doc["decisions"]; hasRootDecisions {
		t.Fatalf("expected deploy output to keep decisions under routing, got root-level decisions in:\n%s", string(deployedYAML))
	}

	routing := requireRoutingSection(t, doc)
	requireFirstDecisionName(t, routing, "new_route")
	if projections, ok := routing["projections"].(map[string]any); ok && len(projections) > 0 {
		t.Fatalf("expected replace deploy to drop omitted routing.projections, got %#v", projections)
	}
}

func writeDeployTestBaseConfig(t *testing.T) string {
	t.Helper()

	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	baseCfg := minimalDeployTestConfig("old_route")
	baseCfg.Projections.Partitions = []config.ProjectionPartition{{
		Name:        "legacy_partition",
		Semantics:   "softmax_exclusive",
		Temperature: 0.1,
		Members:     []string{"math"},
		Default:     "math",
	}}
	baseYAML := mustMarshalCanonicalConfigYAML(t, baseCfg)
	if err := os.WriteFile(configPath, baseYAML, 0o644); err != nil {
		t.Fatalf("write base config: %v", err)
	}
	return configPath
}

func executeRouterConfigUpdateAndRead(
	t *testing.T,
	method string,
	configPath string,
	decisionName string,
) (map[string]any, []byte) {
	t.Helper()

	payloadYAML := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig(decisionName))
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(payloadYAML)})
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	apiServer := &ClassificationAPIServer{configPath: configPath}
	req := httptest.NewRequest(method, "/config/router", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	switch method {
	case http.MethodPatch:
		apiServer.handleConfigPatch(rr, req)
	case http.MethodPut:
		apiServer.handleConfigPut(rr, req)
	default:
		t.Fatalf("unsupported update method %q", method)
	}

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read updated config: %v", err)
	}

	var doc map[string]any
	if err := yaml.Unmarshal(data, &doc); err != nil {
		t.Fatalf("yaml.Unmarshal updated config: %v", err)
	}

	return doc, data
}

func requireRoutingSection(t *testing.T, doc map[string]any) map[string]any {
	t.Helper()

	routing, ok := doc["routing"].(map[string]any)
	if !ok {
		t.Fatalf("expected routing block in updated config, got %#v", doc)
	}
	return routing
}

func requireRoutingProjectionBlock(t *testing.T, routing map[string]any, rawYAML []byte) {
	t.Helper()

	projections, ok := routing["projections"].(map[string]any)
	if !ok || len(projections) == 0 {
		t.Fatalf("expected merge to preserve existing routing.projections, got %#v in:\n%s", routing["projections"], string(rawYAML))
	}
}

func requireFirstDecisionName(t *testing.T, routing map[string]any, want string) {
	t.Helper()

	decisions, ok := routing["decisions"].([]any)
	if !ok || len(decisions) == 0 {
		t.Fatalf("expected routing.decisions in updated config, got %#v", routing["decisions"])
	}
	firstDecision, ok := decisions[0].(map[string]any)
	if !ok {
		t.Fatalf("expected routing.decisions[0] to be a map, got %#v", decisions[0])
	}
	if got := firstDecision["name"]; got != want {
		t.Fatalf("expected routing.decisions[0].name = %s, got %v", want, got)
	}
}

func minimalDeployTestConfig(decisionName string) *config.RouterConfig {
	return &config.RouterConfig{
		APIServer: config.APIServer{
			Listeners: []config.Listener{{
				Name:    "http-8899",
				Address: "0.0.0.0",
				Port:    8899,
				Timeout: "120s",
			}},
		},
		BackendModels: config.BackendModels{
			DefaultModel: "qwen-test",
			ModelConfig: map[string]config.ModelParams{
				"qwen-test": {},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name:     "primary",
				Address:  "vllm",
				Port:     8000,
				Protocol: "http",
				Model:    "qwen-test",
				Weight:   1,
			}},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				Categories: []config.Category{{
					CategoryMetadata: config.CategoryMetadata{Name: "math"},
				}},
			},
			Decisions: []config.Decision{{
				Name:     decisionName,
				Priority: 100,
				Tier:     1,
				Rules: config.RuleCombination{
					Operator: "AND",
					Conditions: []config.RuleNode{{
						Type: "domain",
						Name: "math",
					}},
				},
				ModelRefs: []config.ModelRef{{
					Model: "qwen-test",
				}},
			}},
		},
	}
}

func mustMarshalCanonicalConfigYAML(t *testing.T, cfg *config.RouterConfig) []byte {
	t.Helper()
	canonical := config.CanonicalConfigFromRouterConfig(cfg)
	data, err := yaml.Marshal(canonical)
	if err != nil {
		t.Fatalf("yaml.Marshal canonical config: %v", err)
	}
	return data
}
