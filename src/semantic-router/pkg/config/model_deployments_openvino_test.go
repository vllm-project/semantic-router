package config

import (
	"strings"
	"testing"
)

func TestOpenVINODeploymentContract(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{"ir": {Provider: "openvino", Artifact: "models/export", Device: "CPU", Input: ModelInputBudget{MaxTokens: 64, Overflow: "truncate"}}}
	cfg.GlobalModelBindings = map[string]ModelBinding{"embedding": {Deployment: "ir", Contract: "embedding.v1", Adapter: "bert", Head: "openvino_model.xml"}, "domain_classifier": {Deployment: "ir", Contract: RemoteClassifierContractLabelDistribution, Adapter: "bert"}}
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	global, ok := plan.LookupGlobal("embedding")
	if !ok || global.Deployment.Device != "CPU" {
		t.Fatalf("global=%+v", global)
	}
	for _, test := range []struct {
		name, want string
		change     func(*RouterConfig)
	}{
		{"precision", "native", func(c *RouterConfig) {
			d := c.ModelDeployments["ir"]
			d.Precision = "fp16"
			c.ModelDeployments["ir"] = d
		}},
		{"window", "reject or truncate", func(c *RouterConfig) {
			d := c.ModelDeployments["ir"]
			d.Input.Overflow = "window"
			c.ModelDeployments["ir"] = d
		}},
		{"token head", "sequence label", func(c *RouterConfig) {
			c.GlobalModelBindings["pii_classifier"] = ModelBinding{Deployment: "ir", Contract: RemoteClassifierContractTokenSpans, Adapter: "bert"}
		}},
		{"wrong graph", "XML", func(c *RouterConfig) {
			b := c.GlobalModelBindings["embedding"]
			b.Head = "model.onnx"
			c.GlobalModelBindings["embedding"] = b
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			copy := *cfg
			copy.ModelDeployments = cloneModelMap(cfg.ModelDeployments)
			copy.GlobalModelBindings = cloneModelMap(cfg.GlobalModelBindings)
			test.change(&copy)
			if _, err := CompileModelBindings(&copy); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error=%v", err)
			}
		})
	}
}

func TestOpenVINOLegacyDefaultsBecomeOwnedExecution(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")
	models := EmbeddingModels{EmbeddingConfig: HNSWConfig{Backend: EmbeddingBackendOpenVINO}}
	models.UseCPU = true
	if p, d := DefaultEmbeddingExecution(models); p != "openvino" || d != "CPU" {
		t.Fatalf("%s %s", p, d)
	}
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "openvino")
	if p, d := DefaultCategoryExecution(false); p != "openvino" || d != "AUTO" {
		t.Fatalf("%s %s", p, d)
	}
}
