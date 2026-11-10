package config

import (
	"bytes"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// A dropped classifier deployment takes what named it along, and migrate says
// so at each place: a rule's deployment, a binding of a type or of one rule,
// and its admission gate. The rewritten configuration migrates to itself.
func TestClassifierReferenceMigrationNotes(t *testing.T) {
	migrated, notes, err := MigrateConfig(readGolden(t, filepath.Join("testdata", "migrate", "invalid-value-classifier-provider-references", "before.yaml")))
	if err != nil {
		t.Fatal(err)
	}
	got := map[string][]string{}
	for _, note := range notes {
		got[note.Field] = append(got[note.Field], note.Code+": "+note.Message)
	}
	for path, phrase := range map[string]string{
		"recipes[default].signals.safety[unsafe-content]":       "migration_action: it asked deployment content-safety-binary (provider classify), which is removed",
		"recipes[default].signals.complexity[needs_reasoning]":  "migration_action: it asked deployment difficulty-scorer (provider classify), which is removed",
		"recipes[default].model_bindings.safety.unsafe-privacy": "migration_action: binding safety.unsafe-privacy named deployment content-safety-binary",
		"global.model_catalog.bindings.pii":                     "migration_action: binding pii named deployment pii-spans",
		"global.model_catalog.admission.pii-spans":              "migration_change: the admission gate of deployment pii-spans",
	} {
		if !slices.ContainsFunc(got[path], func(message string) bool { return strings.HasPrefix(message, phrase) }) {
			t.Errorf("%s: notes %q, want one starting %q", path, got[path], phrase)
		}
	}
	again, notes, err := MigrateConfig(migrated)
	if err != nil || len(notes) > 0 || !bytes.Equal(again, migrated) {
		t.Fatalf("migrating the rewritten configuration again: %v %+v\n%s", err, notes, again)
	}
}

// embeddings.semantic naming the embedding model of an earlier release
// migrates to its Vela 1.0 replacement, with a note to re-embed what was
// indexed with it: on the CPU the pre-declared vela-embedding serves it, off
// the CPU a model_runtime deployment that embeddings.default names.
func TestEarlierSemanticEmbeddingModelMigratesToVela(t *testing.T) {
	for _, tc := range []struct {
		useCPU, want string
	}{
		{"true", ""},
		{"false", "deployment: Vela-1.0-Encoder-307M-Embedding-auto"},
	} {
		document := []byte(migrateModels + `recipes:
  - name: default
global:
  model_catalog:
    embeddings:
      semantic:
        mmbert_model_path: models/mmbert-embed-32k-2d-matryoshka
        use_cpu: ` + tc.useCPU + `
        embedding_config: {model_type: mmbert}
`)
		migrated, notes, err := MigrateConfig(document)
		if err != nil {
			t.Fatalf("use_cpu %s: migrate: %v", tc.useCPU, err)
		}
		if bytes.Contains(migrated, []byte("mmbert")) {
			t.Errorf("use_cpu %s: the earlier model is still named:\n%s", tc.useCPU, migrated)
		}
		if tc.want == "" && bytes.Contains(migrated, []byte("embeddings")) {
			t.Errorf("use_cpu %s: want the pre-declared vela-embedding, got\n%s", tc.useCPU, migrated)
		}
		if tc.want != "" && (!bytes.Contains(migrated, []byte(tc.want)) || !bytes.Contains(migrated, []byte("artifact: vllm-sr/Vela-1.0-Encoder-307M-Embedding"))) {
			t.Errorf("use_cpu %s: want embeddings.default %q on Vela Embedding, got\n%s", tc.useCPU, tc.want, migrated)
		}
		if !slices.ContainsFunc(notesAt(notes)["global.model_catalog.embeddings.semantic"], func(message string) bool {
			return strings.Contains(message, "models/Vela-1.0-Encoder-307M-Embedding") && strings.Contains(message, "re-embedded")
		}) {
			t.Errorf("use_cpu %s: no note names the replacement and the re-embedding: %+v", tc.useCPU, notes)
		}
		if _, loadErr := ParseYAMLBytesWithoutEnvExpansion(migrated); loadErr != nil {
			t.Errorf("use_cpu %s: the migrated file does not load: %v", tc.useCPU, loadErr)
		}
		again, more, err := MigrateConfig(migrated)
		if err != nil || len(more) > 0 || !bytes.Equal(again, migrated) {
			t.Errorf("use_cpu %s: migrating the rewritten file again: %v %+v\n%s", tc.useCPU, err, more, again)
		}
	}
}

// An empty global.model_catalog.external written as a mapping is a removed
// section like the empty list the golden removed-section-external-models-empty
// covers: the load names it with the migrate hint, and migrate deletes it.
func TestEmptyExternalModelsMigrate(t *testing.T) {
	document := []byte(migrateModels + `recipes:
  - name: default
global:
  model_catalog:
    external: {}
`)
	_, err := ParseYAMLBytesWithoutEnvExpansion(document)
	if err == nil || !strings.Contains(err.Error(), "global.model_catalog.external: removed_section: external models are removed; run `vllm-sr config migrate`") {
		t.Errorf("external: {} loads with %v", err)
	}
	migrated, _, err := MigrateConfig(document)
	if err != nil || bytes.Contains(migrated, []byte("model_catalog")) {
		t.Errorf("external: {} migrates with %v to\n%s", err, migrated)
	}
}
