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

// migrateModels is the provider catalog of the migrate tests below: two
// described models.
const migrateModels = `version: v0.5
providers:
  defaults: {model: small}
  models:
    - name: small
      backend_refs: [{name: small, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
      description: Fast general model
    - name: large
      backend_refs: [{name: large, endpoint: "127.0.0.1:8001", protocol: http, weight: 1}]
      description: Strong reasoning model
`

// notesAt are a migration's notes by the place they name.
func notesAt(notes []ConfigWarning) map[string][]string {
	got := map[string][]string{}
	for _, note := range notes {
		got[note.Field] = append(got[note.Field], note.Message)
	}
	return got
}

func wantNotes(t *testing.T, notes []ConfigWarning, want map[string][]string) {
	t.Helper()
	got := notesAt(notes)
	for path, phrases := range want {
		for _, phrase := range phrases {
			if !slices.ContainsFunc(got[path], func(message string) bool { return strings.Contains(message, phrase) }) {
				t.Errorf("%s: notes %q, want one with %q", path, got[path], phrase)
			}
		}
	}
}

// A reask rule matched a streak: the current turn repeated each of its
// lookback_turns previous user turns in a row, and then a rule of fewer
// lookback turns did not match. A turn_relation rule has no streak, so
// migrate says so on each rule that looked back further than one turn, and
// names the rules the rewrite leaves asking the same question.
func TestReaskMigrationNotesTheLostStreak(t *testing.T) {
	_, notes, err := MigrateConfig([]byte(migrateModels + `recipes:
  - name: default
    signals:
      reasks:
        - {name: likely_dissatisfied, threshold: 0.8, lookback_turns: 1}
        - {name: persistently_dissatisfied, threshold: 0.8, lookback_turns: 2}
        - {name: repeated, threshold: 0.9}
`))
	if err != nil {
		t.Fatal(err)
	}
	wantNotes(t, notes, map[string][]string{
		"recipes[default].signals.reasks[persistently_dissatisfied]": {
			"v0.5 has no streak: lookback_turns 2 matched only when the current turn repeated each of the 2 previous user turns",
			`input: {previous_turns: 4}`,
		},
		"recipes[default].signals.reasks[repeated]": {"turn_relation rule repeated asks what likely_dissatisfied asks"},
	})
	if got := notesAt(notes)["recipes[default].signals.reasks[likely_dissatisfied]"]; len(got) > 0 {
		t.Errorf("a one-turn rule kept its meaning, but migrate notes %q", got)
	}
}

// turn_relation reads at most 32 previous turns: a reask rule that looked
// back further migrates to 32, with a note, where the rewrite used to fail.
func TestReaskMigrationCapsPreviousTurns(t *testing.T) {
	migrated, notes, err := MigrateConfig([]byte(migrateModels + `recipes:
  - name: default
    signals:
      reasks:
        - {name: long, lookback_turns: 17}
        - {name: longer, lookback_turns: 20}
`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(migrated), "previous_turns: 32") || strings.Contains(string(migrated), "previous_turns: 34") {
		t.Fatalf("migrate wrote\n%s\nwant previous_turns 32", migrated)
	}
	wantNotes(t, notes, map[string][]string{
		"recipes[default].signals.reasks[long]":   {"lookback_turns 17 is 34 turns; a turn_relation rule reads at most 32"},
		"recipes[default].signals.reasks[longer]": {"turn_relation rule longer asks what long asks"},
	})
}

// A v0.5 user_feedbacks or modality rule without labels matches nothing
// itself, whatever it is called, and its conditions name the option as their
// label: it loads, and migrate leaves it alone. A rule no condition names
// with a label is read as before v0.5, its name choosing its option.
func TestLabelLessRuleNamedForAnOptionLoads(t *testing.T) {
	document := []byte(migrateModels + `recipes:
  - name: default
    signals:
      user_feedbacks: [{name: satisfied}]
      modality: [{name: AR}]
    decisions:
      - name: happy
        priority: 100
        rules: {type: user_feedback, name: satisfied, label: SAT}
        modelRefs: [{model: small}]
      - name: text
        priority: 90
        rules: {type: modality, name: AR, label: text}
        modelRefs: [{model: small}]
`)
	if _, err := ParseYAMLBytesWithoutEnvExpansion(document); err != nil {
		t.Fatalf("a v0.5 rule without labels does not load: %v", err)
	}
	migrated, notes, err := MigrateConfig(document)
	if err != nil || len(notes) > 0 || !bytes.Equal(migrated, document) {
		t.Fatalf("migrate rewrote a v0.5 configuration: %v %+v\n%s", err, notes, migrated)
	}
}

// A contrastive jailbreak rule's threshold bounded the gap between its
// closest jailbreak and benign patterns, not a probability: migrate keeps the
// value as the predicate and says to re-fit it.
func TestContrastiveJailbreakThresholdNotesARefit(t *testing.T) {
	_, notes, err := MigrateConfig([]byte(migrateModels + `recipes:
  - name: default
    signals:
      jailbreak:
        - {name: injection, method: contrastive, threshold: 0.1, jailbreak_patterns: [ignore previous instructions], benign_patterns: [explain the policy]}
        - {name: attack, threshold: 0.7}
`))
	if err != nil {
		t.Fatal(err)
	}
	wantNotes(t, notes, map[string][]string{
		"recipes[default].signals.jailbreak[injection]": {"re-fit predicate.gte 0.1"},
	})
	if got := notesAt(notes)["recipes[default].signals.jailbreak[attack]"]; len(got) > 0 {
		t.Errorf("a classifier threshold kept its meaning, but migrate notes %q", got)
	}
}

// A v0.5 domain rule whose name the load suggests a domain for is a likely
// typo: the load keeps the suggestion, and migrate drops neither the rule nor
// the decision that reads it, so it refuses with the same suggestion.
func TestDomainNearMissKeepsTheSuggestion(t *testing.T) {
	for _, near := range [][2]string{{"maths", "math"}, {"computer_science", "computer science"}} {
		document := []byte(migrateModels + `recipes:
  - name: default
    signals:
      domains: [{name: ` + near[0] + `}]
    decisions:
      - name: near
        priority: 100
        rules: {type: domain, name: ` + near[0] + `}
        modelRefs: [{model: small}]
`)
		suggestion := `the rule's name "` + near[0] + `" is not a domain`
		_, err := ParseYAMLBytesWithoutEnvExpansion(document)
		if err == nil || !strings.Contains(err.Error(), suggestion) || !strings.Contains(err.Error(), `did you mean "`+near[1]+`"?`) || strings.Contains(err.Error(), "vllm-sr config migrate") {
			t.Errorf("%s loads with %v, want the suggestion %q alone", near[0], err, near[1])
		}
		if _, _, err = MigrateConfig(document); err == nil || !strings.Contains(err.Error(), suggestion) {
			t.Errorf("%s migrates with %v, want the suggestion", near[0], err)
		}
	}
}

// Before v0.5 a projection partition synthesized its default member when no
// member matched, so a domain rule named for no domain matched as the
// default. Migrate keeps it, and the decision on it, with labels [other], and
// says what a member that is no partition's default now matches.
func TestPartitionDomainRuleIsKept(t *testing.T) {
	migrated, notes, err := MigrateConfig([]byte(migrateModels + `recipes:
  - name: default
    signals:
      domains: [{name: math}, {name: general}, {name: chitchat}]
    projections:
      partitions: [{name: subjects, semantics: exclusive, members: [math, general, chitchat], default: general}]
    decisions:
      - name: general
        priority: 100
        rules: {type: domain, name: general}
        modelRefs: [{model: small}]
`))
	if err != nil {
		t.Fatal(err)
	}
	if want := "domains: [{name: math}, {name: general, labels: [other]}, {name: chitchat, labels: [other]}]"; !strings.Contains(string(migrated), want) ||
		!strings.Contains(string(migrated), "rules: {type: domain, name: general}") {
		t.Fatalf("migrate wrote\n%s\nwant %s and the decision on general", migrated, want)
	}
	wantNotes(t, notes, map[string][]string{
		"recipes[default].signals.domains[general]":  {"matched only as the default of projection partition subjects", "so it matches as before"},
		"recipes[default].signals.domains[chitchat]": {"it now matches when the decision model chooses other"},
	})
}
