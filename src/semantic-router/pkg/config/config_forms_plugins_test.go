package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const pluginFormsDoc = `version: v0.3
routing:
  decisions:
    - name: guarded
      plugins:
PLUGINS
global:
  model_catalog:
    modules:
      prompt_guard: {on_error: ON_ERROR, on_unscanned: ON_UNSCANNED}
`

// migratePlugins rewrites the plugin forms of a decision whose plugins are
// written in plugins, and returns the decision's plugins and the notes.
func migratePlugins(t *testing.T, plugins, onError, onUnscanned string) ([]map[string]interface{}, []ConfigWarning) {
	t.Helper()
	doc := strings.NewReplacer("PLUGINS", plugins, "ON_ERROR", onError, "ON_UNSCANNED", onUnscanned).Replace(pluginFormsDoc)
	migrated, notes, err := migrateConfigForms([]byte(doc), pluginForms)
	if err != nil {
		t.Fatalf("migrate: %v", err)
	}
	var out struct {
		Routing struct {
			Decisions []struct {
				Plugins []map[string]interface{} `yaml:"plugins"`
			} `yaml:"decisions"`
		} `yaml:"routing"`
	}
	if err := yaml.Unmarshal(migrated, &out); err != nil {
		t.Fatal(err)
	}
	return out.Routing.Decisions[0].Plugins, notes
}

func noteSaying(notes []ConfigWarning, code, text string) bool {
	for _, note := range notes {
		if note.Code == code && strings.Contains(note.Message, text) {
			return true
		}
	}
	return false
}

func TestMigrateRemovesDisabledPlugins(t *testing.T) {
	plugins, notes := migratePlugins(t, `        - {type: response_jailbreak, configuration: {enabled: false}}
        - {type: hallucination, configuration: {hallucination_action: body}}
        - {type: masking, configuration: {enabled: false}}`, "allow", "block")
	if len(plugins) != 0 {
		t.Fatalf("plugins = %v, want the disabled plugins removed", plugins)
	}
	for _, plugin := range []string{"response_jailbreak", "hallucination", "masking"} {
		if !noteSaying(notes, migrationChangeCode, "the disabled "+plugin+" plugin is removed") {
			t.Errorf("no note says the disabled %s plugin is removed: %+v", plugin, notes)
		}
	}
}

// P1: hallucination_action: block annotated the answer, and the rule keeps
// that, with a note to write action: block.
func TestMigrateHallucinationBlockKeepsWhatItDid(t *testing.T) {
	plugins, notes := migratePlugins(t, `        - {type: hallucination, configuration: {enabled: true, hallucination_action: block, unverified_factual_action: none}}`, "allow", "block")
	guard := plugins[0]["configuration"].(map[string]interface{})
	rule := guard["rules"].([]interface{})[0].(map[string]interface{})
	if plugins[0]["type"] != DecisionPluginResponseGuard || rule["action"] != ResponseGuardAnnotate || guard["on_unknown"] != ResponseGuardNone {
		t.Fatalf("plugins = %v, want an annotating rule with on_unknown none", plugins)
	}
	if !noteSaying(notes, migrationActionCode, "never blocked") {
		t.Fatalf("notes = %+v, want one saying block never blocked", notes)
	}
}

// A decision's existing response_guard takes the rewritten rule, and the
// stricter on_unknown.
func TestMigrateJoinsTheDecisionsResponseGuard(t *testing.T) {
	plugins, _ := migratePlugins(t, `        - type: response_guard
          configuration:
            rules: [{when: {type: keyword, name: probe}, action: annotate}]
            on_unknown: none
        - {type: response_jailbreak, configuration: {enabled: true, action: block}}`, "block", "block")
	if len(plugins) != 1 {
		t.Fatalf("plugins = %v, want one response_guard", plugins)
	}
	guard := plugins[0]["configuration"].(map[string]interface{})
	if rules := guard["rules"].([]interface{}); len(rules) != 2 || guard["on_unknown"] != ResponseGuardBlock {
		t.Fatalf("guard = %v, want both rules and on_unknown block", guard)
	}
}

// An answer the scan could not read counted as clean only when prompt_guard
// allowed both failures and unscanned content.
func TestMigrateResponseJailbreakUnknownFollowsPromptGuard(t *testing.T) {
	for _, tc := range []struct {
		onError, onUnscanned, want string
	}{
		{"allow", "allow", ResponseGuardNone},
		{"allow", "block", ResponseGuardBlock},
		{"block", "allow", ResponseGuardBlock},
	} {
		plugins, _ := migratePlugins(t, `        - {type: response_jailbreak, configuration: {enabled: true, action: block}}`, tc.onError, tc.onUnscanned)
		if got := plugins[0]["configuration"].(map[string]interface{})["on_unknown"]; got != tc.want {
			t.Errorf("on_error %s, on_unscanned %s: on_unknown = %v, want %s", tc.onError, tc.onUnscanned, got, tc.want)
		}
	}
}

func TestMigrateMaskingMovesItsThresholdOntoItsRule(t *testing.T) {
	doc := `version: v0.3
routing:
  signals:
    pii: [{name: personal}]
  decisions:
    - name: support
      plugins:
        - {type: masking, configuration: {enabled: true, signal: personal, threshold: 0.7}}
`
	migrated, _, err := migrateConfigForms([]byte(doc), pluginForms)
	if err != nil {
		t.Fatal(err)
	}
	cfg := string(migrated)
	if !strings.Contains(cfg, "pii: [{name: personal, threshold: 0.7}]") || strings.Contains(cfg, "signal: personal, threshold") {
		t.Fatalf("migrated:\n%s\nwant the threshold on pii rule personal", cfg)
	}
}
