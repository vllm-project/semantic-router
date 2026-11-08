package config

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// migratedSummary migrates a document and summarizes what it resolves to,
// failing the test when the migration keeps a deprecated field.
func migratedSummary(t *testing.T, name string, data []byte) (resolutionSummary, string) {
	t.Helper()
	migrated, notes, err := MigrateModelResources(data)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	for _, note := range notes {
		if note.Code == warningDeprecatedModelField {
			t.Errorf("%s: the migration kept a deprecated field %+v:\n%s", name, note, migrated)
		}
	}
	cfg, err := ParseYAMLBytesWithoutEnvExpansion(migrated)
	if err != nil {
		t.Fatalf("%s: the migrated configuration does not load: %v\n%s", name, err, migrated)
	}
	return summarizeResolution(cfg), string(migrated)
}

// answered drops the signals no model answers: a module that ran no model
// and one switched off run none either way.
func answered(signals map[string]string) map[string]string {
	out := map[string]string{}
	for key, model := range signals {
		if model != "none" {
			out[key] = model
		}
	}
	return out
}

func TestMigratedDeprecatedFixturesResolveAsBefore(t *testing.T) {
	files, err := filepath.Glob(filepath.Join(deprecatedFixtures, "*.yaml"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no fixtures: %v", err)
	}
	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			data, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			before := summarizeFile(file)
			after, migrated := migratedSummary(t, file, data)
			if !reflect.DeepEqual(answered(before.Signals), answered(after.Signals)) || !reflect.DeepEqual(before.Started, after.Started) || before.Error != after.Error {
				t.Errorf("the migrated configuration resolves differently:\n was %+v\n now %+v\n%s", before, after, migrated)
			}
		})
	}
}

func TestMigratingTheEarlierFormsWritesTheirReplacement(t *testing.T) {
	examples := readDocExamples(t, filepath.Join(docsRoot(), "docs", "installation", "model-resource-migration.md"))
	data, err := os.ReadFile(modelCatalogDocsBase)
	if err != nil {
		t.Fatal(err)
	}
	var base map[string]interface{}
	if err = yaml.Unmarshal(data, &base); err != nil {
		t.Fatal(err)
	}
	for _, example := range examples {
		if !strings.HasSuffix(example.title, "-before") {
			continue
		}
		t.Run(example.title, func(t *testing.T) {
			var overlay map[string]interface{}
			if err := yaml.Unmarshal([]byte(example.text), &overlay); err != nil {
				t.Fatal(err)
			}
			document, err := yaml.Marshal(mergeYAML(base, overlay))
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := ParseYAMLBytesWithoutEnvExpansion(document)
			if err != nil {
				t.Fatal(err)
			}
			before := summarizeResolution(cfg)
			after, migrated := migratedSummary(t, example.title, document)
			if !reflect.DeepEqual(answered(before.Signals), answered(after.Signals)) || !reflect.DeepEqual(before.Started, after.Started) {
				t.Errorf("the migrated configuration resolves differently:\n was %+v\n now %+v\n%s", before, after, migrated)
			}
		})
	}
}

// The model runtime's migration page shows what `vllm-sr config migrate`
// writes: the CLI's own step produces testdata/model_runtime_migration
// (checked by the CLI's docs test), and the Router's step must turn that into
// the page's legacy.migrated.yaml, in both languages.
func TestModelRuntimeMigrationExampleIsWhatConfigMigrateWrites(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "model_runtime_migration", "legacy.engine.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	migrated, _, err := MigrateConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	var got interface{}
	if err = yaml.Unmarshal(migrated, &got); err != nil {
		t.Fatal(err)
	}
	page := filepath.Join("model-runtime", "migrate.md")
	for _, path := range []string{
		filepath.Join(docsRoot(), "docs", page),
		filepath.Join(docsRoot(), "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", page),
	} {
		found := false
		for _, example := range readDocExamples(t, path) {
			if example.title != "legacy.migrated.yaml" {
				continue
			}
			found = true
			var documented interface{}
			if err := yaml.Unmarshal([]byte(example.text), &documented); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, documented) {
				t.Errorf("%s: legacy.migrated.yaml is not what config migrate writes:\n%s", path, migrated)
			}
		}
		if !found {
			t.Errorf("%s has no legacy.migrated.yaml block", path)
		}
	}
}

func TestMigrationKeepsACanonicalDocumentAsWritten(t *testing.T) {
	document := []byte("version: v0.5\nrecipes: [{name: default}]\nglobal:\n  model_catalog:\n    bindings: {pii: primary}\n")
	migrated, kept, err := MigrateModelResources(document)
	if err != nil || len(kept) > 0 || string(migrated) != string(document) {
		t.Fatalf("a canonical document changed: %q %+v %v", migrated, kept, err)
	}
}

// memoryRewriteDocument declares the memory rewrite model as an external[]
// chat model with the given extra settings.
func memoryRewriteDocument(settings string) []byte {
	return []byte(`version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
recipes:
  - name: default
    signals:
      keywords: [{name: greeting, operator: OR, keywords: [hello]}]
    decisions:
      - name: route
        priority: 100
        rules: {operator: OR, conditions: [{type: keyword, name: greeting}]}
        modelRefs: [{model: general}]
global:
  model_catalog:
    external:
      - name: rewriter
        llm_provider: vllm
        model_role: memory_rewrite
        llm_endpoint: {address: llm.example.com, port: 8000, protocol: http}
        llm_model_name: qwen3
` + settings)
}

func TestMigrationMovesAKeyWrittenInTheConfigurationToTheEnvironment(t *testing.T) {
	migrated, notes, err := MigrateModelResources(memoryRewriteDocument("        access_key: sk-test-written-inline\n"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(migrated), "sk-test-written-inline") || !strings.Contains(string(migrated), "api_key_env: VLLM_SR_REWRITER_API_KEY") {
		t.Fatalf("the key was not moved to an environment variable:\n%s", migrated)
	}
	moved := false
	for _, note := range notes {
		moved = moved || (note.Code == migrationActionCode && strings.Contains(note.Message, "VLLM_SR_"))
	}
	if !moved {
		t.Fatalf("no note says which environment variable to set: %+v", notes)
	}
}

func TestMigrationCarriesAChatModelsGeneration(t *testing.T) {
	migrated, _, err := MigrateModelResources(memoryRewriteDocument("        max_tokens: 64\n        temperature: 0.2\n"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(migrated), "max_tokens: 64") || !strings.Contains(string(migrated), "temperature: 0.2") || strings.Contains(string(migrated), "external:") {
		t.Fatalf("the chat model's generation was not carried:\n%s", migrated)
	}
}
