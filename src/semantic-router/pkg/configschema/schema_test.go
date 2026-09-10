package configschema

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func repositoryRoot(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve schema test path")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "../../../.."))
}

func TestGeneratedArtifactsMatchGoSource(t *testing.T) {
	root := repositoryRoot(t)
	want, err := GenerateFromSource(root)
	if err != nil {
		t.Fatal(err)
	}
	artifacts := []string{
		filepath.Join(root, "src", "semantic-router", "pkg", "configschema", "router-config-v0.3.schema.json"),
	}
	for _, artifact := range artifacts {
		got, readErr := os.ReadFile(artifact)
		if readErr != nil {
			t.Fatalf("read %s: %v", artifact, readErr)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("generated config schema is stale: %s; run go generate ./pkg/configschema", artifact)
		}
	}
	if !bytes.Equal(Document(), want) {
		t.Fatal("embedded schema does not match generated Go source")
	}
	wantTypeScript, err := GenerateTypeScriptContract(want)
	if err != nil {
		t.Fatal(err)
	}
	typeScriptPath := filepath.Join(root, "dashboard", "frontend", "src", "generated", "routerConfigContract.ts")
	gotTypeScript, err := os.ReadFile(typeScriptPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotTypeScript, wantTypeScript) {
		t.Fatalf("generated config contract is stale: %s; run go generate ./pkg/configschema", typeScriptPath)
	}
}

func TestCanonicalSchemaIsTheOnlyTrackedSchemaArtifact(t *testing.T) {
	root := repositoryRoot(t)
	for _, obsolete := range []string{
		filepath.Join(root, "config", "schemas", "router-config-v0.3.schema.json"),
		filepath.Join(root, "src", "vllm-sr", "cli", "config_schema", "router-config-v0.3.schema.json"),
		filepath.Join(root, "dashboard", "frontend", "src", "generated", "router-config-v0.3.schema.json"),
	} {
		if _, err := os.Stat(obsolete); err == nil {
			t.Errorf("obsolete schema mirror still exists: %s", obsolete)
		} else if !os.IsNotExist(err) {
			t.Errorf("inspect obsolete schema mirror %s: %v", obsolete, err)
		}
	}
}

func TestSchemaPublishesEveryRoutingSurface(t *testing.T) {
	var document struct {
		Properties map[string]json.RawMessage `json:"properties"`
		Extension  schemaExtension            `json:"x-vllm-sr"`
	}
	if err := json.Unmarshal(Document(), &document); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"version", "listeners", "providers", "evaluation", "routing", "entrypoints", "recipes", "global", "setup"} {
		if _, ok := document.Properties[field]; !ok {
			t.Errorf("root schema is missing %q", field)
		}
	}
	if document.Extension.ContractVersion != ContractVersion {
		t.Fatalf("contract version = %q", document.Extension.ContractVersion)
	}
	if got, want := len(document.Extension.Signals), len(routerconfig.SignalCatalog()); got != want {
		t.Errorf("signal surfaces = %d, want %d", got, want)
	}
	if got, want := len(document.Extension.Algorithms), len(routerconfig.DecisionAlgorithmCatalog()); got != want {
		t.Errorf("algorithm surfaces = %d, want %d", got, want)
	}
	if got, want := len(document.Extension.Plugins), len(routerconfig.DecisionPluginCatalog()); got != want {
		t.Errorf("plugin surfaces = %d, want %d", got, want)
	}
	if got, want := len(document.Extension.ProjectionInputTypes), len(routerconfig.SupportedProjectionInputTypes()); got != want {
		t.Errorf("projection input types = %d, want %d", got, want)
	}
	if got, want := len(document.Extension.Projections), len(routerconfig.ProjectionCatalog()); got != want {
		t.Errorf("projection surfaces = %d, want %d", got, want)
	}
	if len(document.Extension.GlobalSections) == 0 {
		t.Fatal("global section catalog is empty")
	}
	globalPaths := map[string]bool{}
	for _, surface := range document.Extension.GlobalSections {
		if surface.Key == "" || surface.Layer == "" || surface.DisplayName == "" || len(surface.Path) == 0 {
			t.Errorf("incomplete global section surface: %#v", surface)
		}
		path := strings.Join(surface.Path, ".")
		if globalPaths[path] {
			t.Errorf("duplicate global section path: %s", path)
		}
		globalPaths[path] = true
	}
	for _, path := range []string{
		"router",
		"services.management_api",
		"services.startup_status",
		"model_catalog.kbs",
		"model_catalog.admission",
		"model_catalog.modules.complexity",
	} {
		if !globalPaths[path] {
			t.Errorf("global section catalog is missing %q", path)
		}
	}
	for _, surface := range document.Extension.Signals {
		if surface.Type == "" || surface.Collection == "" || surface.SchemaRef == "" {
			t.Errorf("incomplete signal surface: %#v", surface)
		}
	}
	for _, surface := range document.Extension.Plugins {
		if surface.Type == "" || surface.SchemaRef == "" {
			t.Errorf("incomplete plugin surface: %#v", surface)
		}
	}
}

func TestSchemaMapsCustomScalarTypesToTheirPublicYAMLShape(t *testing.T) {
	type scalarBranch struct {
		Type    string      `json:"type"`
		Const   interface{} `json:"const"`
		Minimum json.Number `json:"minimum"`
	}
	var document struct {
		Definitions map[string]struct {
			Properties map[string]struct {
				OneOf []scalarBranch `json:"oneOf"`
			} `json:"properties"`
		} `json:"$defs"`
	}
	if err := json.Unmarshal(Document(), &document); err != nil {
		t.Fatal(err)
	}
	budget, ok := document.Definitions["ContextCompressionBudgetConfig"]
	if !ok {
		t.Fatal("ContextCompressionBudgetConfig schema is missing")
	}
	limit, ok := budget.Properties["trigger_tokens"]
	if !ok || len(limit.OneOf) != 2 {
		t.Fatalf("CompressionTokenLimit schema = %#v, want integer-or-auto scalar", limit)
	}
	if limit.OneOf[0].Type != "integer" || limit.OneOf[0].Minimum.String() != "0" {
		t.Fatalf("numeric CompressionTokenLimit branch = %#v", limit.OneOf[0])
	}
	if limit.OneOf[1].Type != "string" || limit.OneOf[1].Const != "auto" {
		t.Fatalf("auto CompressionTokenLimit branch = %#v", limit.OneOf[1])
	}
}
