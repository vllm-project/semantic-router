package config

import (
	"strings"
	"testing"
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
			document: decisionPluginsDocument("        - {type: shadow_dispatch, configuration: {enabled: true, model: c, sample_rate: 0.1}}\n"),
			want:     []string{"type: router_replay", "enabled: true", "shadow:", "model: c", "sample_rate: 0.1"},
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

func notesContain(notes []ConfigWarning, text string) bool {
	for _, note := range notes {
		if strings.Contains(note.Message, text) {
			return true
		}
	}
	return false
}
