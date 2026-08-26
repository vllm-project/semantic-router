package dashboard

import (
	"bytes"
	"errors"
	"io"
	"os"
	"testing"

	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
)

func TestDashboardManagedGatewayTargetsPrivateBackendDispatch(t *testing.T) {
	documents := readManagedInferenceGatewayDocuments(t)
	assertManagedInferenceBackend(t, documents["Backend"])
	assertManagedAIServiceBackend(t, documents["AIServiceBackend"])
	assertManagedAIGatewayRoute(t, documents["AIGatewayRoute"])
	if resourceManifests[len(resourceManifests)-1] != managedInferenceGatewayManifest {
		t.Fatal("managed gateway overlay must replace the generic route after its prerequisites")
	}
}

func readManagedInferenceGatewayDocuments(t *testing.T) map[string]map[string]interface{} {
	t.Helper()
	raw, err := os.ReadFile("managed-inference-gateway.yaml")
	if err != nil {
		t.Fatal(err)
	}
	decoder := utilyaml.NewYAMLOrJSONDecoder(bytes.NewReader(raw), 4096)
	documents := make(map[string]map[string]interface{})
	for {
		var document map[string]interface{}
		err := decoder.Decode(&document)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		kind, _ := document["kind"].(string)
		if kind == "" {
			t.Fatal("managed inference manifest contains a document without a Kind")
		}
		documents[kind] = document
	}
	return documents
}

func assertManagedInferenceBackend(t *testing.T, document map[string]interface{}) {
	t.Helper()
	backend := requireNestedMap(t, document, "spec")
	endpoints := requireNestedSlice(t, backend, "endpoints")
	if len(endpoints) != 1 {
		t.Fatalf("managed Backend endpoints = %d, want 1", len(endpoints))
	}
	fqdn := requireNestedMap(t, requireMap(t, endpoints[0]), "fqdn")
	if got := fqdn["hostname"]; got != "semantic-router-backend-dispatch.vllm-semantic-router-system.svc.cluster.local" {
		t.Fatalf("managed Backend hostname = %v", got)
	}
	if got := fqdn["port"]; got != float64(8180) {
		t.Fatalf("managed Backend port = %v, want 8180", got)
	}
}

func assertManagedAIServiceBackend(t *testing.T, document map[string]interface{}) {
	t.Helper()
	serviceBackend := requireNestedMap(t, document, "spec", "backendRef")
	if got := serviceBackend["name"]; got != "semantic-router-backend-dispatch" {
		t.Fatalf("managed AIServiceBackend target = %v", got)
	}
}

func assertManagedAIGatewayRoute(t *testing.T, document map[string]interface{}) {
	t.Helper()
	route := requireNestedMap(t, document, "spec")
	rules := requireNestedSlice(t, route, "rules")
	if len(rules) != 1 {
		t.Fatalf("managed AIGatewayRoute rules = %d, want 1 catch-all", len(rules))
	}
	if _, matched := requireMap(t, rules[0])["matches"]; matched {
		t.Fatal("managed AIGatewayRoute must dispatch every authorized logical model")
	}
	backendRefs := requireNestedSlice(t, requireMap(t, rules[0]), "backendRefs")
	if len(backendRefs) != 1 || requireMap(t, backendRefs[0])["name"] != "semantic-router-backend-dispatch" {
		t.Fatalf("managed AIGatewayRoute backendRefs = %#v", backendRefs)
	}
}

func requireNestedMap(t *testing.T, value map[string]interface{}, path ...string) map[string]interface{} {
	t.Helper()
	current := value
	for _, key := range path {
		next, ok := current[key]
		if !ok {
			t.Fatalf("missing manifest field %q", key)
		}
		current = requireMap(t, next)
	}
	return current
}

func requireNestedSlice(t *testing.T, value map[string]interface{}, key string) []interface{} {
	t.Helper()
	items, ok := value[key].([]interface{})
	if !ok {
		t.Fatalf("manifest field %q is not a list", key)
	}
	return items
}

func requireMap(t *testing.T, value interface{}) map[string]interface{} {
	t.Helper()
	mapped, ok := value.(map[string]interface{})
	if !ok {
		t.Fatalf("manifest value is not an object: %#v", value)
	}
	return mapped
}
