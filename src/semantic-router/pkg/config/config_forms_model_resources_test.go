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
