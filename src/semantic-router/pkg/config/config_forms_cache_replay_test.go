package config

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func decisionPluginsDocument(plugins string) string {
	return "recipes:\n  - name: default\n    decisions:\n      - name: d\n        plugins:\n" + strings.ReplaceAll(plugins, "        - ", "          - ")
}

// The response cache, shadow and retention rewrites keep what each old form
// did, and refuse a document that writes the old form beside the new one.
func TestCacheReplayRetentionMigrations(t *testing.T) {
	cases := []struct {
		name     string
		forms    []ConfigForm
		document string
		want     []string
		wantErr  string
		wantNote string
	}{
		{
			name:     "an alias beside response_cache",
			forms:    responseCacheForms,
			document: decisionPluginsDocument("        - {type: semantic_cache, configuration: {enabled: true}}\n        - {type: response_cache, configuration: {enabled: true}}\n"),
			wantErr:  "also has a response_cache plugin",
		},
		{
			name:     "the semantic threshold",
			forms:    responseCacheForms,
			document: decisionPluginsDocument("        - {type: response_cache, configuration: {enabled: true, semantic: {similarity_threshold: 0.9}}}\n"),
			want:     []string{"match:", "recall:", "top_k: 3", "min_similarity: 0.9"},
		},
		{
			name:     "both thresholds",
			forms:    responseCacheForms,
			document: decisionPluginsDocument("        - {type: response_cache, configuration: {similarity_threshold: 0.8, semantic: {similarity_threshold: 0.9}}}\n"),
			wantErr:  "keep one",
		},
		{
			name:     "a disabled request control",
			forms:    responseCacheForms,
			document: decisionPluginsDocument("        - {type: response_cache, configuration: {enabled: true, allow_request_controls: false}}\n"),
			want:     []string{"enabled: true}"},
		},
		{
			name:     "a shadow without router_replay",
			forms:    routerReplayForms,
			document: decisionPluginsDocument("        - {type: shadow_dispatch, configuration: {enabled: true, model: c, sample_rate: 0.1, forward_headers: [x-tenant]}}\n"),
			want:     []string{"type: router_replay", "enabled: true", "shadow:", "models: [c]", "sample_rate: 0.1", "forward_headers: [x-tenant]"},
		},
		{
			name:     "integer-valued shadow limits",
			forms:    routerReplayForms,
			document: decisionPluginsDocument("        - {type: shadow_dispatch, configuration: {enabled: true, model: c, max_concurrency: 2.0, max_queue_depth: 4.0, max_capture_bytes: 4096.0}}\n"),
			want:     []string{"max_concurrency: 2", "max_queue_size: 4", "max_bytes: 4096"},
		},
		{
			name:     "shadow timeout above the new ceiling",
			forms:    routerReplayForms,
			document: decisionPluginsDocument("        - {type: shadow_dispatch, configuration: {enabled: true, model: c, timeout_seconds: 600}}\n"),
			want:     []string{"timeout: 300s"},
			wantNote: "timeout_seconds is limited to 300s",
		},
		{
			name:     "shadow byte limit below the new minimum",
			forms:    routerReplayForms,
			document: decisionPluginsDocument("        - {type: shadow_dispatch, configuration: {enabled: true, model: c, max_response_bytes: 1}}\n"),
			want:     []string{"max_response_bytes: 2"},
			wantNote: "max_response_bytes is raised to 2",
		},
		{
			name:     "fractional shadow limits remain invalid",
			forms:    routerReplayForms,
			document: decisionPluginsDocument("        - {type: shadow_dispatch, configuration: {enabled: true, model: c, max_concurrency: 2.5}}\n"),
			wantErr:  "must be a non-negative integer",
		},
		{
			name:     "a disabled shadow",
			forms:    routerReplayForms,
			document: decisionPluginsDocument("        - {type: system_prompt, configuration: {enabled: true}}\n        - {type: shadow_dispatch, configuration: {enabled: false, model: c}}\n"),
			want:     []string{"system_prompt"},
		},
		{
			name:     "a shadow on a disabled router_replay",
			forms:    routerReplayForms,
			document: decisionPluginsDocument("        - {type: router_replay, configuration: {enabled: false}}\n        - {type: shadow_dispatch, configuration: {enabled: true, model: c}}\n"),
			want:     []string{"shadow:"},
			wantNote: "enable it to keep the shadow",
		},
		{
			name:     "a second shadow",
			forms:    routerReplayForms,
			document: decisionPluginsDocument("        - {type: router_replay, configuration: {shadow: {model: b}}}\n        - {type: shadow_dispatch, configuration: {enabled: true, model: c}}\n"),
			wantErr:  "already has a shadow",
		},
		{
			name:     "emits of another kind",
			forms:    retentionForms,
			document: "recipes:\n  - name: default\n    decisions:\n      - name: d\n        emits: [{kind: log}]\n",
			wantErr:  `emits kind "log" has no replacement`,
		},
		{
			name:     "emits beside retention",
			forms:    retentionForms,
			document: "recipes:\n  - name: default\n    decisions:\n      - name: d\n        retention: {drop: true}\n        emits: [{kind: retention, retention: {ttl_turns: 2}}]\n",
			wantErr:  "keep retention",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			migrated, notes, err := migrateConfigForms([]byte(tc.document), tc.forms)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error = %v, want containing %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range tc.want {
				if !strings.Contains(string(migrated), want) {
					t.Fatalf("migrated document lacks %q:\n%s", want, migrated)
				}
			}
			for _, removed := range []string{"shadow_dispatch", "similarity_threshold", "allow_request_controls", "semantic_cache"} {
				if strings.Contains(string(migrated), removed) {
					t.Fatalf("migrated document still writes %s:\n%s", removed, migrated)
				}
			}
			if tc.wantNote != "" && !notesContain(notes, tc.wantNote) {
				t.Fatalf("notes %+v lack %q", notes, tc.wantNote)
			}
		})
	}
}

func TestResponseCacheAliasesAreNoPluginTypes(t *testing.T) {
	for _, alias := range responseCacheAliases {
		if IsSupportedDecisionPluginType(alias) {
			t.Fatalf("%s still resolves to a plugin type", alias)
		}
	}
}

func TestShadowDispatchMigrationNullIntegerDefaults(t *testing.T) {
	for _, field := range []string{"max_concurrency", "max_queue_depth", "timeout_seconds", "max_response_bytes", "max_capture_bytes"} {
		t.Run(field, func(t *testing.T) {
			document := decisionPluginsDocument(fmt.Sprintf("        - {type: shadow_dispatch, configuration: {enabled: true, model: c, %s: null}}\n", field))
			migrated, _, err := migrateConfigForms([]byte(document), routerReplayForms)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(migrated), "null") || strings.Contains(string(migrated), field) {
				t.Fatalf("null integer should select the default:\n%s", migrated)
			}
			if !strings.Contains(string(migrated), "router_replay") || !strings.Contains(string(migrated), "models: [c]") {
				t.Fatalf("shadow was lost:\n%s", migrated)
			}
		})
	}
}

func TestShadowDispatchMigrationYAMLMerge(t *testing.T) {
	cases := []struct {
		name, configuration, want, wantErr string
	}{
		{"inline model", "{enabled: true, <<: {model: c}, max_concurrency: 2}", "{models: [c], execution: {max_concurrency: 2}}", ""},
		{"inherited enabled", "{<<: {enabled: true, model: c, max_queue_depth: 4}}", "{models: [c], execution: {max_queue_size: 4}}", ""},
		{"explicit override", "{model: c, max_concurrency: 3, <<: {enabled: true, model: ignored, max_concurrency: -1}}", "{models: [c], execution: {max_concurrency: 3}}", ""},
		{"merge sequence precedence", "{<<: [{enabled: true, model: c, max_concurrency: 2}, {model: ignored, max_concurrency: 9, timeout_seconds: 6}]}", "{models: [c], execution: {max_concurrency: 2, timeout: 6s}}", ""},
		{"nested alias merge", "{<<: [&base {enabled: true, model: c}, {<<: *base, timeout_seconds: 6}]}", "{models: [c], execution: {timeout: 6s}}", ""},
		{"inherited null", "{enabled: true, <<: {model: c, max_concurrency: null}}", "{models: [c]}", ""},
		{"inherited negative", "{enabled: true, <<: {model: c, max_concurrency: -1}}", "", "must be a non-negative integer"},
		{"inherited fraction", "{enabled: true, <<: {model: c, timeout_seconds: 2.5}}", "", "must be a non-negative integer"},
		{"inherited unknown", "{enabled: true, <<: {model: c, unexpected: true}}", "", "unknown shadow field"},
		{"quoted merge is a field", "{enabled: true, model: c, '<<': {max_concurrency: 2}}", "", "unknown shadow field"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			document := decisionPluginsDocument("        - {type: shadow_dispatch, configuration: " + tc.configuration + "}\n")
			migrated, _, err := migrateConfigForms([]byte(document), routerReplayForms)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error = %v, want containing %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var doc yaml.Node
			if err := yaml.Unmarshal(migrated, &doc); err != nil {
				t.Fatal(err)
			}
			recipe := sequenceAt(documentRoot(&doc), "recipes")[0]
			decision := sequenceAt(recipe, "decisions")[0]
			plugin := pluginOfType(mappingValue(decision, "plugins"), DecisionPluginRouterReplay)
			shadow := mappingValue(mappingValue(plugin, "configuration"), "shadow")
			if shadow == nil {
				t.Fatalf("shadow was lost:\n%s", migrated)
			}
			var got, want map[string]interface{}
			if err := shadow.Decode(&got); err != nil {
				t.Fatal(err)
			}
			if err := yaml.Unmarshal([]byte(tc.want), &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("shadow = %#v, want %#v", got, want)
			}
		})
	}
}

func notesContain(notes []ConfigWarning, text string) bool {
	for _, note := range notes {
		if strings.Contains(note.Message, text) {
			return true
		}
	}
	return false
}
