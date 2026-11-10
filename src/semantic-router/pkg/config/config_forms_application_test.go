package config

import (
	"bytes"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// A safety rule's hazard part becomes the harm categories it names, and
// migrate says what does not carry over: the unsafe threshold, a category
// that names no harm category, and a rule left with the unsafe verdict alone.
// The migrated rules migrate to themselves.
func TestSafetyHazardMigrationNotes(t *testing.T) {
	migrated, notes, err := MigrateConfig(readGolden(t, filepath.Join("testdata", "migrate", "removed-safety-hazard", "before.yaml")))
	if err != nil {
		t.Fatal(err)
	}
	got := map[string][]string{}
	for _, note := range notes {
		got[note.Field] = append(got[note.Field], note.Message)
	}
	for name, phrases := range map[string][]string{
		"unsafe-privacy": {
			"hazard categories privacy are the harm categories PII/Privacy, S7",
			"the rule's threshold 0.5 has no counterpart",
		},
		"violent": {
			"hazard categories violence, weapons are the harm categories Violence, Threat, S1, Guns and Illegal Weapons, S9",
			"the rule's threshold 0.4 has no counterpart",
		},
		"abuse": {
			"hazard categories Harassment, threat are the harm categories Harassment, Threat",
			"the rule's threshold 0.5 has no counterpart",
			"hazard categories spam name no harm category; name the ones they meant in labels",
		},
		"spam": {
			"hazard categories spam name no harm category, so the rule matches an unsafe request as its threshold did",
		},
	} {
		path := "recipes[default].signals.safety[" + name + "]"
		if len(got[path]) != len(phrases) {
			t.Errorf("%s: notes %q, want %d", path, got[path], len(phrases))
		}
		for _, phrase := range phrases {
			if !slices.ContainsFunc(got[path], func(message string) bool { return strings.Contains(message, phrase) }) {
				t.Errorf("%s: notes %q, want one with %q", path, got[path], phrase)
			}
		}
	}
	again, notes, err := MigrateConfig(migrated)
	if err != nil || len(notes) > 0 || !bytes.Equal(again, migrated) {
		t.Fatalf("migrating the rewritten rules again: %v %+v\n%s", err, notes, again)
	}
}

// A kb rule never matches after migrate: the notes say what each dropped
// rule, decision, condition and score input did and how to restore it.
func TestKBMigrationNotes(t *testing.T) {
	migrated, notes, err := MigrateConfig(readGolden(t, filepath.Join("testdata", "migrate", "removed-kb-signal", "before.yaml")))
	if err != nil {
		t.Fatal(err)
	}
	got := map[string][]string{}
	for _, note := range notes {
		got[note.Field] = append(got[note.Field], note.Code+": "+note.Message)
	}
	for path, phrase := range map[string]string{
		"recipes[default].signals.kb[privacy]": "migration_action: kb rule privacy is dropped: no v0.5 signal reads a knowledge base, so it never matches; to route on its labels, ask " +
			`signals.systemone: [{name: privacy, question: {type: set, instructions: "Which of these does the request concern?", labels: [{key: pii}, {key: internal_document}]}}]`,
		"recipes[default].decisions[private]":                 "migration_action: the decision is dropped: its rules need kb rule privacy, and no v0.5 signal reads a knowledge base",
		"recipes[default].decisions[urgent_or_private].rules": "migration_change: dropped from its rules: kb rule privacy, since no v0.5 signal reads a knowledge base",
		"recipes[default].decisions[urgent_public].rules":     "migration_change: dropped from its rules: kb rule privacy",
		"recipes[default].projections.scores[policy_pressure]": "migration_action: score policy_pressure is dropped: it read only the kb_metric private_vs_public input, " +
			"and no v0.5 signal reads a knowledge base",
		"recipes[default].decisions[policy_heavy]":      "migration_action: the decision is dropped: its rules need projection output policy_heavy, and score policy_pressure is dropped",
		"recipes[default].projections.scores[pressure]": "migration_action: the kb privacy input is dropped: no v0.5 signal reads a knowledge base; re-fit the bands",
	} {
		if !slices.ContainsFunc(got[path], func(message string) bool { return strings.HasPrefix(message, phrase) }) {
			t.Errorf("%s: notes %q, want one starting %q", path, got[path], phrase)
		}
	}
	again, notes, err := MigrateConfig(migrated)
	if err != nil || len(notes) > 0 || !bytes.Equal(again, migrated) {
		t.Fatalf("migrating the migrated kb rules again: %v %+v\n%s", err, notes, again)
	}
}

// A v0.3 pii rule with source: tool_result read the request's tool results
// and nothing else, include_history included. Migrated, it still leaves the
// user turn and the earlier turns out, so the harness instructions in an
// agent's turn never decide it.
func TestAToolResultPIIRuleKeepsReadingOnlyTheToolResults(t *testing.T) {
	migrated, _, err := MigrateConfig(readGolden(t, filepath.Join("testdata", "migrate", "removed-application-fields", "before.yaml")))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseYAMLBytesWithoutEnvExpansion(migrated)
	if err != nil {
		t.Fatal(err)
	}
	rule, ok := cfg.consumerScopes()[DefaultRecipeName].ApplicationRuleByName(SignalTypePII, "personal")
	if !ok {
		t.Fatal("the pii rule is gone")
	}
	if input := rule.Rule.Input; input.ReadsTurn() || input.History || !input.ToolResults {
		t.Fatalf("the migrated rule reads %+v, want the tool results alone", input)
	}
}
