package agentnative

import (
	"encoding/json"
	"errors"
	"reflect"
	"sort"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestConfigCatalogCoversAuthoritativeSurfaceWithoutRuntimeState(t *testing.T) {
	catalog, err := NewConfigCatalog()
	if err != nil {
		t.Fatal(err)
	}
	expected := map[ComponentKind][]string{
		ComponentSignal:     config.SupportedSignalTypes(),
		ComponentProjection: {"mapping", "partition", "score"},
		ComponentDecision:   {"decision"},
		ComponentAlgorithm:  config.SupportedDecisionAlgorithmTypes(),
		ComponentPlugin:     config.SupportedDecisionPluginTypes(),
	}
	for kind, names := range expected {
		sort.Strings(names)
		if got := catalogNames(t, catalog, kind); !reflect.DeepEqual(got, names) {
			t.Fatalf("%s catalog names = %v, want %v", kind, got, names)
		}
		for _, name := range names {
			page, describeErr := catalog.Describe(CatalogQuery{Kind: kind, Name: name, PageSize: 1})
			if describeErr != nil {
				t.Fatalf("Describe(%s, %s): %v", kind, name, describeErr)
			}
			if len(page.Data) != 1 || len(page.Data[0].Schema) == 0 {
				t.Fatalf("Describe(%s, %s) returned no focused schema", kind, name)
			}
			assertModelSafeSchema(t, kind, name, page.Data[0].Schema)
		}
	}
}

func TestConfigCatalogRejectsNameWithoutKind(t *testing.T) {
	catalog, err := NewConfigCatalog()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := catalog.Describe(CatalogQuery{Name: "static"}); !errors.Is(err, agentmanagement.ErrInvalid) {
		t.Fatalf("Describe(name without kind) error = %v, want ErrInvalid", err)
	}
}

func TestCatalogAndExamplePagesEncodeEmptyDataAsArrays(t *testing.T) {
	catalog := &ConfigCatalog{revision: "sha256:test", descriptors: []ComponentDescriptor{}}
	catalogPage, err := catalog.Describe(CatalogQuery{})
	if err != nil {
		t.Fatal(err)
	}
	assertEmptyJSONArray(t, catalogPage)

	examples := &DistributionExamples{revision: "sha256:test", items: []RecipeExample{}}
	examplePage, err := examples.List(ExampleQuery{})
	if err != nil {
		t.Fatal(err)
	}
	assertEmptyJSONArray(t, examplePage)
}

func catalogNames(t *testing.T, catalog *ConfigCatalog, kind ComponentKind) []string {
	t.Helper()
	var result []string
	cursor := ""
	for {
		page, err := catalog.Describe(CatalogQuery{Kind: kind, PageSize: maximumCatalogPageSize, Cursor: cursor})
		if err != nil {
			t.Fatalf("Describe(%s): %v", kind, err)
		}
		for _, descriptor := range page.Data {
			if len(descriptor.Schema) != 0 {
				t.Fatalf("browse descriptor %s/%s exposed a schema", kind, descriptor.Name)
			}
			result = append(result, descriptor.Name)
		}
		if !page.HasMore {
			break
		}
		if page.NextCursor == "" {
			t.Fatalf("Describe(%s) hasMore without nextCursor", kind)
		}
		cursor = page.NextCursor
	}
	sort.Strings(result)
	return result
}

func assertModelSafeSchema(t *testing.T, kind ComponentKind, name string, raw json.RawMessage) {
	t.Helper()
	var schema any
	if err := json.Unmarshal(raw, &schema); err != nil {
		t.Fatalf("decode %s/%s schema: %v", kind, name, err)
	}
	properties := map[string]struct{}{}
	meta := map[string][]string{}
	collectSchemaNames(schema, properties, meta)
	for _, forbidden := range []string{
		"id", "revision", "resource_id", "resource_revision", "catalog_revision",
		"provider_catalog_revision", "model", "models", "modelRefs", "model_refs",
		"synthesis_model", "analysis_models", "analysis_overrides", "model_path",
		"models_path", "pretrained_path", "embedding_model_ref", "tools_db_path",
		"backend_config", "backends", "connection", "connections", "credential_id",
		"provider_credential_id", "provider_model_id", "api_key", "base_url", "endpoint",
		"address", "password",
	} {
		if _, found := properties[forbidden]; found {
			t.Errorf("%s/%s schema exposed forbidden authoring property %q", kind, name, forbidden)
		}
	}
	if _, found := meta["$id"]; found {
		t.Errorf("%s/%s schema exposed a schema identity", kind, name)
	}
	for _, reference := range meta["$ref"] {
		if len(reference) < len("#/$defs/") || reference[:len("#/$defs/")] != "#/$defs/" {
			t.Errorf("%s/%s schema exposed non-local reference %q", kind, name, reference)
		}
	}
}

func collectSchemaNames(value any, properties map[string]struct{}, meta map[string][]string) {
	switch typed := value.(type) {
	case map[string]any:
		collectSchemaObject(typed, properties, meta)
	case []any:
		for _, nested := range typed {
			collectSchemaNames(nested, properties, meta)
		}
	}
}

func collectSchemaObject(value map[string]any, properties map[string]struct{}, meta map[string][]string) {
	for key, nested := range value {
		if fields, ok := schemaProperties(key, nested); ok {
			for field := range fields {
				properties[field] = struct{}{}
			}
		}
		if len(key) > 0 && key[0] == '$' {
			text, _ := nested.(string)
			meta[key] = append(meta[key], text)
		}
		collectSchemaNames(nested, properties, meta)
	}
}

func schemaProperties(key string, value any) (map[string]any, bool) {
	if key != "properties" {
		return nil, false
	}
	fields, ok := value.(map[string]any)
	return fields, ok
}

func assertEmptyJSONArray(t *testing.T, value any) {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &root); err != nil {
		t.Fatal(err)
	}
	if string(root["data"]) != "[]" {
		t.Fatalf("data = %s, want []", root["data"])
	}
}
