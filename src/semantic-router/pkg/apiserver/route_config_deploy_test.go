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

func TestHandleConfigDeployMergesCanonicalRoutingShape(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")

	baseYAML := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("old_route"))
	if err := os.WriteFile(configPath, baseYAML, 0o644); err != nil {
		t.Fatalf("write base config: %v", err)
	}

	deployYAML := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("new_route"))
	body, err := json.Marshal(ConfigDeployRequest{YAML: string(deployYAML)})
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	apiServer := &ClassificationAPIServer{configPath: configPath}
	req := httptest.NewRequest(http.MethodPost, "/config/deploy", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	apiServer.handleConfigDeploy(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	deployedYAML, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read deployed config: %v", err)
	}

	var doc map[string]any
	if err := yaml.Unmarshal(deployedYAML, &doc); err != nil {
		t.Fatalf("yaml.Unmarshal deployed config: %v", err)
	}

	if _, hasRootDecisions := doc["decisions"]; hasRootDecisions {
		t.Fatalf("expected deploy output to keep decisions under routing, got root-level decisions in:\n%s", string(deployedYAML))
	}

	routing, ok := doc["routing"].(map[string]any)
	if !ok {
		t.Fatalf("expected routing block in deployed config, got %#v", doc)
	}
	decisions, ok := routing["decisions"].([]any)
	if !ok || len(decisions) == 0 {
		t.Fatalf("expected routing.decisions in deployed config, got %#v", routing["decisions"])
	}
	firstDecision, ok := decisions[0].(map[string]any)
	if !ok {
		t.Fatalf("expected routing.decisions[0] to be a map, got %#v", decisions[0])
	}
	if got := firstDecision["name"]; got != "new_route" {
		t.Fatalf("expected routing.decisions[0].name = new_route, got %v", got)
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
