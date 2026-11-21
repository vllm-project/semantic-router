package config

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestRouterReplayShadowConfigDefaults(t *testing.T) {
	c := RouterReplayShadowConfig{Models: []string{" candidate ", "other"}}.WithDefaults()
	if c.Models[0] != "candidate" || *c.Execution.MaxConcurrency != 2 || *c.Execution.MaxQueueSize != 8 || c.Execution.Timeout != "30s" || *c.Budget.MaxResponseBytes != 2<<20 || *c.Capture.MaxBytes != 8192 || c.EffectiveSampleRate() != 1 {
		t.Fatalf("unexpected defaults: %+v", c)
	}
	original := RouterReplayShadowConfig{Models: []string{" candidate "}}
	_ = original.WithDefaults()
	if original.Models[0] != " candidate " {
		t.Fatal("defaults mutated source models")
	}
}

func TestRouterReplayShadowPayload(t *testing.T) {
	for _, tc := range []struct{ name, body, want string }{
		{"missing models", `{}`, "models is required"},
		{"legacy", `{"model":"a"}`, "unknown field"},
		{"blank", `{"models":[" "]}`, "empty"},
		{"duplicate", `{"models":["a"," a "]}`, "duplicate"},
		{"fractional", `{"budget":{"max_response_bytes":2.5}}`, "cannot unmarshal"},
		{"budget type", `{"budget":"bad"}`, "cannot unmarshal"},
		{"unknown budget field", `{"budget":{"max_tokens":1}}`, "unknown field"},
		{"unknown execution field", `{"execution":{"extra":1}}`, "unknown field"},
		{"unknown capture field", `{"capture":{"extra":1}}`, "unknown field"},
		{"invalid timeout", `{"execution":{"timeout":"0s"}}`, "timeout"},
		{"empty timeout", `{"execution":{"timeout":""}}`, "timeout"},
		{"null timeout", `{"execution":{"timeout":null}}`, "timeout"},
		{"null execution", `{"execution":null}`, "execution"},
		{"null budget", `{"budget":null}`, "budget"},
		{"null capture", `{"capture":null}`, "capture"},
		{"allowlisted header", `{"forward_headers":["x-tenant"]}`, ""},
		{"credential header", `{"forward_headers":["Authorization"]}`, "credential header"},
		{"pseudo-header", `{"forward_headers":[":authority"]}`, "pseudo-header"},
		{"timeout exceeds maximum", `{"execution":{"timeout":"5m1ms"}}`, "timeout"},
		{"aggregate bytes too low", `{"models":["a","b"],"budget":{"max_response_bytes":3}}`, "two bytes"},

		{"integer floats", `{"execution":{"max_concurrency":2.0,"max_queue_size":8.0},"budget":{"max_response_bytes":4096.0},"capture":{"max_bytes":4096.0}}`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var fields map[string]interface{}
			if err := json.Unmarshal([]byte(tc.body), &fields); err != nil {
				t.Fatal(err)
			}
			if tc.name != "missing models" && fields["models"] == nil {
				fields["models"] = []string{"a"}
			}
			payload := MustStructuredPayload(map[string]interface{}{"enabled": true, "shadow": fields})
			_, err := DecodeDecisionPluginAt(PluginAt{Decision: "route", Type: DecisionPluginRouterReplay}, DecisionPlugin{Type: DecisionPluginRouterReplay, Configuration: payload})
			if tc.want == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want %s", err, tc.want)
			}
		})
	}
}

func TestRouterReplayShadowCanonicalRecipe(t *testing.T) {
	for _, tc := range []struct{ name, candidate, algorithm, want string }{
		{"trimmed models and null limits", " candidate ", "static", ""},
		{"unknown second candidate", "unknown", "static", `router_replay shadow model "unknown" has no configured backend`},
		{"looper", "candidate", "ratings", "executor algorithm"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := fmt.Sprintf(`{
			  "version":"v0.5",
			  "providers":{"defaults":{"model":"primary"},"models":[
			    {"name":"primary","backend_refs":[{"endpoint":"127.0.0.1:8000","provider":"vllm"}]},
			    {"name":"candidate","backend_refs":[{"endpoint":"127.0.0.1:8001","provider":"vllm"}]}
			  ]},
			  "recipes":[{"name":"private","decisions":[{
			    "name":"shadow-route", "rules":{"operator":"AND","conditions":[]},
			    "modelRefs":[{"model":"primary"}], "algorithm":{"type":%q},
			    "plugins":[{"type":"router_replay","configuration":{"enabled":true,"shadow":{
			      "models":[" primary ",%q],"sample_rate":null,
			      "execution":{"max_concurrency":null,"max_queue_size":null},
			      "budget":{"max_response_bytes":null},"capture":{"max_bytes":null}
			    }}}]
			  }]}]
			}`, tc.algorithm, tc.candidate)
			cfg, err := ParseYAMLBytes([]byte(body))
			if tc.want != "" {
				if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("got %v, want %s", err, tc.want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, recipe := range cfg.Recipes {
				if recipe.Name == "private" {
					shadow := recipe.Profile.Decisions[0].GetRouterReplayConfig().Shadow.WithDefaults()
					if strings.Join(shadow.Models, ",") != "primary,candidate" || shadow.Execution.Timeout != "30s" || *shadow.Execution.MaxConcurrency != 2 || *shadow.Execution.MaxQueueSize != 8 || *shadow.Budget.MaxResponseBytes != 2<<20 || *shadow.Capture.MaxBytes != 8192 || shadow.EffectiveSampleRate() != 1 {
						t.Fatalf("unexpected parsed defaults: %+v", shadow)
					}
					return
				}
			}
			t.Fatal("private recipe missing")
		})
	}
}

func TestRouterReplayShadowValidateSharedWithConverter(t *testing.T) {
	zero := 0
	c := RouterReplayShadowConfig{Models: []string{"m"}, Budget: ShadowBudgetConfig{MaxResponseBytes: &zero}}
	if err := c.Validate(); err == nil {
		t.Fatal("explicit zero must fail")
	}
	var absent *RouterReplayShadowConfig
	if err := absent.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestShadowCredentialHeaders(t *testing.T) {
	for _, name := range []string{"Authorization", "x-api-key", "Cookie", "x-user-openai-key"} {
		if !IsShadowCredentialHeader(name) {
			t.Errorf("not recognized: %s", name)
		}
	}
	if IsShadowCredentialHeader("traceparent") {
		t.Fatal("traceparent is not a credential")
	}
}

func TestRouterReplayShadowValidatesAllCandidatesInEveryLayer(t *testing.T) {
	for _, layer := range []string{"global", "recipe", "decision"} {
		for _, target := range []string{"candidate", "unknown", "delegated", "caller"} {
			t.Run(layer+"/"+target, func(t *testing.T) {
				var doc map[string]interface{}
				if err := json.Unmarshal([]byte(`{"version":"v0.5","providers":{"defaults":{"model":"primary"},"models":[
				{"name":"primary","backend_refs":[{"endpoint":"127.0.0.1:8000"}]},
				{"name":"candidate","backend_refs":[{"endpoint":"127.0.0.1:8001"}]},
				{"name":"delegated","backend_refs":[{"endpoint":"127.0.0.1:8002","auth":{"credential":"delegated"}}]}
				]},"global":{"services":{"authz":{"delegated_credential":{"enabled":true}}}},
				"recipes":[{"name":"default","decisions":[{"name":"chat","rules":{"operator":"AND","conditions":[]},"modelRefs":[{"model":"primary"}]}]}]}`), &doc); err != nil {
					t.Fatal(err)
				}
				if target == "caller" {
					providers := doc["providers"].(map[string]interface{})
					providers["models"] = []map[string]interface{}{shadowCallerBackend("primary"), shadowCallerBackend("caller")}
				}
				plugin := map[string]interface{}{"type": "router_replay", "configuration": map[string]interface{}{"enabled": true, "shadow": map[string]interface{}{"models": []string{"primary", target}}}}
				recipe := doc["recipes"].([]interface{})[0].(map[string]interface{})
				owner := doc["global"].(map[string]interface{})
				if layer == "recipe" {
					owner = recipe
				}
				if layer == "decision" {
					owner = recipe["decisions"].([]interface{})[0].(map[string]interface{})
				}
				owner["plugins"] = []interface{}{plugin}
				body, err := json.Marshal(doc)
				if err != nil {
					t.Fatal(err)
				}
				_, err = ParseYAMLBytes(body)
				switch target {
				case "candidate":
					if err != nil {
						t.Fatal(err)
					}
				case "unknown":
					if err == nil || !strings.Contains(err.Error(), `model "unknown" has no configured backend`) {
						t.Fatalf("unknown second candidate: %v", err)
					}
				case "delegated", "caller":
					if err == nil || !strings.Contains(err.Error(), "auth.credential: "+target) || !strings.Contains(err.Error(), "call it as the Router") {
						t.Fatalf("%s shadow candidate: %v", target, err)
					}
				}
			})
		}
	}
}

func shadowCallerBackend(name string) map[string]interface{} {
	return map[string]interface{}{
		"name": name, "provider_model_id": "claude-haiku-5-5", "api_format": "anthropic",
		"backend_refs": []map[string]interface{}{{
			"name": "anthropic", "provider": "anthropic", "endpoint": "https://api.anthropic.com",
			"auth": map[string]string{"credential": CredentialCaller},
		}},
	}
}
