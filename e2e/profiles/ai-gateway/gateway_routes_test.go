package aigateway

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestSelectedModelsHaveGatewayBackendRoutes(t *testing.T) {
	config := profileConfig(t)
	defaults := profileMap(t, profileMap(t, config, "providers"), "defaults")
	selected := map[string]bool{defaults["model"].(string): true}
	routings := []map[string]any{profileMap(t, config, "routing")}
	for _, raw := range config["recipes"].([]any) {
		routings = append(routings, profileMap(t, raw.(map[string]any), "routing"))
	}
	for _, routing := range routings {
		for _, raw := range routing["decisions"].([]any) {
			refs, _ := raw.(map[string]any)["modelRefs"].([]any)
			for _, ref := range refs {
				model := ref.(map[string]any)
				name, _ := model["lora_name"].(string)
				if name == "" {
					name = model["model"].(string)
				}
				selected[name] = true
			}
		}
	}

	routes, backends := profileGatewayRoutes(t)
	for model := range selected {
		targets := routes[model]
		if len(targets) == 0 {
			t.Errorf("selected model %q has no exact x-ai-eg-model Gateway route", model)
		}
		for _, target := range targets {
			if !backends[target] {
				t.Errorf("selected model %q routes to undeclared AIServiceBackend %q", model, target)
			}
		}
	}
}

func profileGatewayRoutes(t *testing.T) (map[string][]string, map[string]bool) {
	t.Helper()
	routes := make(map[string][]string)
	backends := make(map[string]bool)
	for _, name := range resourceManifests {
		data, err := os.ReadFile(filepath.Join("..", "..", "..", name))
		if err != nil {
			t.Fatal(err)
		}
		decoder := yaml.NewDecoder(strings.NewReader(string(data)))
		for {
			var resource map[string]any
			if err := decoder.Decode(&resource); errors.Is(err, io.EOF) {
				break
			} else if err != nil {
				t.Fatalf("decode %s: %v", name, err)
			}
			if resource["kind"] == "AIServiceBackend" {
				metadata := profileMap(t, resource, "metadata")
				backends[gatewayResourceNamespace(metadata, "default")+"/"+metadata["name"].(string)] = true
			}
			if resource["kind"] == "AIGatewayRoute" {
				collectGatewayModelRoutes(t, resource, routes)
			}
		}
	}
	return routes, backends
}

func collectGatewayModelRoutes(t *testing.T, resource map[string]any, routes map[string][]string) {
	t.Helper()
	spec := profileMap(t, resource, "spec")
	namespace := gatewayResourceNamespace(profileMap(t, resource, "metadata"), "default")
	if namespace != "default" {
		return
	}
	attached := false
	for _, raw := range spec["parentRefs"].([]any) {
		parent := raw.(map[string]any)
		if parent["name"] == "semantic-router" && parent["kind"] == "Gateway" &&
			parent["group"] == "gateway.networking.k8s.io" && gatewayResourceNamespace(parent, namespace) == "default" {
			attached = true
		}
	}
	if !attached {
		return
	}
	for _, raw := range spec["rules"].([]any) {
		rule := raw.(map[string]any)
		var targets []string
		for _, rawBackend := range rule["backendRefs"].([]any) {
			backend := rawBackend.(map[string]any)
			targets = append(targets, gatewayResourceNamespace(backend, namespace)+"/"+backend["name"].(string))
		}
		for _, match := range rule["matches"].([]any) {
			for _, rawHeader := range match.(map[string]any)["headers"].([]any) {
				header := rawHeader.(map[string]any)
				if header["name"] == "x-ai-eg-model" && header["type"] == "Exact" {
					model := header["value"].(string)
					routes[model] = append(routes[model], targets...)
				}
			}
		}
	}
}

func gatewayResourceNamespace(value map[string]any, fallback string) string {
	if namespace, ok := value["namespace"].(string); ok && namespace != "" {
		return namespace
	}
	return fallback
}
