package config

import (
	"strings"
	"testing"
)

const signalsCompressionGlobal = `global:
  plugins:
    - type: context
      configuration:
        enabled: true
        compress:
          signals: {enabled: true, max_tokens: 512, skip_signals: [jailbreak]}
`

// compress.signals in global.plugins compresses what the signals of every
// recipe read, a recipe's own overrides it field by field, and a block with
// only signals leaves every request whole.
func TestSignalsCompressionLayersThroughGlobalAndRecipePlugins(t *testing.T) {
	cfg, err := parseStructure(t, signalsCompressionGlobal+`entrypoints:
  - model_names: [team/general]
    recipe: general
    default: true
  - model_names: [team/code]
    recipe: code
  - model_names: [team/plain]
    recipe: plain
recipes:
  - name: general
`+structureDecision+`  - name: code
    plugins:
      - type: context
        configuration:
          compress:
            signals: {profile: coding}
`+structureDecision+`  - name: plain
    plugins:
      - type: context
        configuration:
          compress:
            signals: {enabled: false}
`+structureDecision)
	if err != nil {
		t.Fatal(err)
	}
	general, _ := cfg.RecipeByName("general")
	if signals := general.SignalsCompression(); signals == nil || signals.MaxTokens != 512 || signals.NormalizedProfile() != PromptCompressionProfileDefault || !signals.SkipSignalsSet()[SignalTypeJailbreak] || signals.SkipSignalsSet()[SignalTypePII] {
		t.Fatalf("general compresses its signals with %+v, want global.plugins' block", signals)
	}
	code, _ := cfg.RecipeByName("code")
	if signals := code.SignalsCompression(); signals == nil || signals.MaxTokens != 512 || signals.Profile != PromptCompressionProfileCoding {
		t.Fatalf("code compresses its signals with %+v, want its coding profile over the global block", signals)
	}
	plain, _ := cfg.RecipeByName("plain")
	if signals := plain.SignalsCompression(); signals != nil {
		t.Fatalf("plain compresses its signals with %+v, want none: it sets enabled: false", signals)
	}
	if plugin := general.Profile.Decisions[0].GetContextConfig(); plugin == nil || plugin.Compress.Signals == nil || plugin.Compress.CompressesRequest() {
		t.Fatalf("a decision inherits the signals block and compresses no request: %+v", plugin)
	}
}

// A block compresses the request unless it holds only signals.
func TestCompressBlockCompressesTheRequestUnlessItHoldsOnlySignals(t *testing.T) {
	signals := &PromptCompressionConfig{Enabled: true, MaxTokens: 512}
	for _, tc := range []struct {
		name     string
		compress *ContextCompressConfig
		want     bool
	}{
		{"none", nil, false},
		{"empty", &ContextCompressConfig{}, true},
		{"signals alone", &ContextCompressConfig{Signals: signals}, false},
		{"signals and a mode", &ContextCompressConfig{Signals: signals, Mode: ContextCompressionModeAuto}, true},
		{"signals and a match", &ContextCompressConfig{Signals: signals, Match: &MatchConfig{Recall: &MatchRecall{Method: MatchRecallBM25}}}, true},
	} {
		if got := tc.compress.CompressesRequest(); got != tc.want {
			t.Errorf("%s: CompressesRequest() = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// A decision compresses no signals: they read the request before a decision
// is chosen. A budget bounds a compressed request, which a block with only
// signals never makes, and the signals name a profile and their tokens.
func TestSignalsCompressionChecks(t *testing.T) {
	decision := func(configuration string) string {
		return `recipes:
  - name: default
    decisions:
      - name: d
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: m}]
        plugins:
          - type: context
            configuration: ` + configuration + "\n"
	}
	global := func(configuration string) string {
		return "global:\n  plugins:\n    - type: context\n      configuration: " + configuration + "\n"
	}
	for _, tc := range []struct{ name, body, want string }{
		{
			"a decision's signals", decision(`{enabled: true, compress: {signals: {enabled: true, max_tokens: 512}}}`),
			"recipes[default].decisions[d].plugins[0].configuration.compress.signals: the routing signals read the request before a decision is chosen; set compress.signals in global.plugins or the recipe's plugins",
		},
		{
			"a budget over signals alone", global(`{enabled: true, compress: {signals: {enabled: true, max_tokens: 512}}, budget: {max_tokens: 6000}}`),
			"global.plugins[context].configuration.budget bounds the request compress produces, and a compress block with only signals leaves the request whole; set compress.mode: auto or remove budget",
		},
		{
			"an unknown profile", global(`{enabled: true, compress: {signals: {enabled: true, profile: finance, max_tokens: 512}}}`),
			`global.plugins[context].configuration.compress.signals.profile: unknown profile "finance" (valid profiles: default, coding, medical, security, multi_turn)`,
		},
		{
			"no tokens to compress to", global(`{enabled: true, compress: {signals: {enabled: true}}}`),
			"global.plugins[context].configuration.compress.signals.max_tokens must be at least 1",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseStructure(t, tc.body); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want %q", err, tc.want)
			}
		})
	}
	if _, err := parseStructure(t, global(`{enabled: true, compress: {signals: {enabled: true, profile: multi-turn, max_tokens: 512}, mode: auto}, budget: {max_tokens: 6000}}`)); err != nil {
		t.Fatalf("signals beside a compressed request: %v", err)
	}
}

// Migrate drops prompt compression that was off, adds the context plugin to
// global.plugins when there is none, and keeps one signals block.
func TestMigratePromptCompression(t *testing.T) {
	const head = `version: v0.5
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
recipes:
  - name: default
global:
  model_catalog:
    modules:
`
	migrated, notes, err := MigrateConfig([]byte(head + "      prompt_compression: {enabled: false, max_tokens: 512}\n"))
	dropped := false
	for _, note := range notes {
		dropped = dropped || strings.Contains(note.Message, "prompt compression was off, so it compressed nothing, and is dropped")
	}
	if err != nil || strings.Contains(string(migrated), "global:") || !dropped {
		t.Fatalf("migrate kept prompt compression that was off: %v %+v\n%s", err, notes, migrated)
	}
	migrated, _, err = MigrateConfig([]byte(head + "      prompt_compression: {enabled: true, max_tokens: 512}\n"))
	want := "global:\n  plugins:\n    - type: context\n      configuration:\n        enabled: true\n        compress:\n          signals: {enabled: true, max_tokens: 512}\n"
	if err != nil || !strings.HasSuffix(string(migrated), want) {
		t.Fatalf("migrate wrote %v\n%s\nwant it to end with\n%s", err, migrated, want)
	}
	_, _, err = MigrateConfig([]byte(head + `      prompt_compression: {enabled: true, max_tokens: 512}
  plugins:
    - type: context
      configuration: {enabled: true, compress: {signals: {enabled: true, max_tokens: 64}}}
`))
	if err == nil || !strings.Contains(err.Error(), "global.plugins' context plugin already compresses what the signals read; keep one") {
		t.Fatalf("migrate merged two signals blocks: %v", err)
	}
}
