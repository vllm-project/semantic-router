package config

import (
	"maps"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// Where a heuristic rewrite cannot keep a meaning exactly, migrate says so at
// the place to review: a density predicate to re-fit, the temporal markers it
// wrote out, a reference that reads the rule's value, and a count of images.
func TestHeuristicMigrationNotes(t *testing.T) {
	want := map[string][]string{
		"removed-events-signal": {"routing.signals.keywords[critical_payment_event]"},
		"removed-structure-signal": {
			"routing.decisions[dense].rules",
			"routing.projections.scores[shape].inputs[constraint_dense]",
			"routing.signals.keywords[constraint_dense].density",
		},
		"removed-actions-signal":             nil,
		"removed-conversation-image-content": {"recipes[gallery].routing.signals.input_modality[many_images]"},
	}
	for name, places := range want {
		_, notes, err := MigrateConfig(readGolden(t, filepath.Join("testdata", "migrate", name, "before.yaml")))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		actions := map[string]string{}
		for _, note := range notes {
			if note.Code == migrationActionCode {
				actions[note.Field] = note.Message
			}
		}
		if got := slices.Sorted(maps.Keys(actions)); !slices.Equal(got, places) {
			t.Errorf("%s: notes at %v, want %v", name, got, places)
		}
	}
}

// A rewritten rule keeps its references naming one rule: a name the target
// collection already holds is refused, in every routing scope.
func TestHeuristicMigrationRefusesNameCollisions(t *testing.T) {
	for name, document := range map[string]string{
		"events": `version: v0.3
routing:
  signals:
    keywords: [{name: urgent, operator: OR, keywords: [now]}]
    events: [{name: urgent, event_types: [outage]}]
`,
		"actions": `version: v0.3
routing:
  signals:
    systemone:
      - {name: action, question: {type: noul, instructions: "Does the request ask for an action?"}}
    actions: [{name: fix}]
`,
		"image_content": `version: v0.3
recipes:
  - name: media
    routing:
      signals:
        input_modality: [{name: pictures, modality: image}]
        conversation:
          - {name: pictures, feature: {type: exists, source: {type: image_content}}}
`,
	} {
		_, _, err := MigrateConfig([]byte(document))
		if err == nil || !strings.Contains(err.Error(), "already has a rule named") {
			t.Errorf("%s: %v", name, err)
		}
	}
}
