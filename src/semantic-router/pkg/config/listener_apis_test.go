package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const listenerAPIsDocument = `version: v0.5
listeners:
  - name: public
    port: 8899
    apis:
%s
providers:
  defaults:
    model: m
  models:
    - name: m
      api_format: %s
      backend_refs:
        - endpoint: 127.0.0.1:8000
          protocol: http
          provider: vllm
recipes:
  - name: default
`

func loadListenerAPIs(t *testing.T, apis, apiFormat string) (*RouterConfig, error) {
	t.Helper()
	return ParseYAMLBytes([]byte(strings.Replace(strings.Replace(listenerAPIsDocument, "%s", apis, 1), "%s", apiFormat, 1)))
}

func TestTheVendorNamesLoadAsTheirAPIs(t *testing.T) {
	cfg, err := loadListenerAPIs(t, "      openai: {models: [m]}\n      anthropic: {}", "openai")
	if err != nil {
		t.Fatal(err)
	}
	listener := cfg.Listeners[0]
	if got := listener.ServedAPIs(); strings.Join(got, ",") != "chat_completions,messages" {
		t.Fatalf("served %v", got)
	}
	if api, _ := listener.API("chat_completions"); !api.Allows("m") || api.Allows("other") {
		t.Fatalf("chat_completions models %v", api.Models)
	}
	if got := cfg.GetModelAPIFormat("m"); got != APIFormatChatCompletions {
		t.Fatalf("api_format openai reads as %q", got)
	}
	cfg, err = loadListenerAPIs(t, "      messages: {}", "anthropic")
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.GetModelAPIFormat("m"); got != APIFormatMessages {
		t.Fatalf("api_format anthropic reads as %q", got)
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(exported), "api_format: messages") || strings.Contains(string(exported), "anthropic") {
		t.Fatalf("export names the alias:\n%s", exported)
	}
}

func TestAnAPINamedTwiceIsRefused(t *testing.T) {
	_, err := loadListenerAPIs(t, "      openai: {}\n      chat_completions: {models: [m]}", "chat_completions")
	if err == nil || !strings.Contains(err.Error(), "listeners[public].apis: duplicate_api: openai and chat_completions name the same API, chat_completions") {
		t.Fatalf("got %v", err)
	}
}

func TestListenerAPIEntriesAreCheckedByTheRegistry(t *testing.T) {
	for apis, want := range map[string]string{
		"      realtime: {}":                                `api "realtime" is not supported; the apis are chat_completions, responses, messages, images, videos, systemone`,
		"      chat_completions: {schema: openai/v1}":       "apis.chat_completions.schema: invalid_value: api chat_completions has one schema, openai/v1, so a listener names none",
		"      chat_completions: {max_deadline_ms: 5}":      "apis.chat_completions.max_deadline_ms: invalid_value: the chat_completions API takes no max_deadline_ms setting",
		"      systemone: {schema: systemone/0.2}":          "a listener serves api systemone in vllm-sr/5.0 or vllm-sr/4.0",
		"      systemone: {schema: vllm-sr/6.0}":            "a listener serves api systemone in vllm-sr/5.0 or vllm-sr/4.0",
		"      systemone: {max_deadline_ms: 0}":             "apis.systemone.max_deadline_ms: invalid_value: must be greater than zero",
		"      chat_completions: {vsr: {redact: false}}":    "apis.chat_completions.vsr: invalid_value: the chat_completions API takes no vsr setting",
		"      systemone: {vsr: {router_goal: cheap}}":      `apis.systemone.vsr.router_goal: invalid_value: "cheap" is not a router goal; use one of quality, balanced, save`,
		"      systemone: {vsr: {pii: false}}":              `unknown field "pii"`,
		"      chat_completions: {models: []}":              "apis.chat_completions.models: invalid_value: lists no model",
		"      chat_completions: {models: [m, m]}":          `lists "m" twice`,
		"      systemone: {models: [not-a-decision-model]}": `public System One model "not-a-decision-model" is not declared`,
		"      chat_completions: {colour: red}":             `unknown field "colour"`,
	} {
		if _, err := loadListenerAPIs(t, apis, "chat_completions"); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s\n got %v\nwant %q", apis, err, want)
		}
	}
	if _, err := loadListenerAPIs(t, "      {}", "chat_completions"); err == nil || !strings.Contains(err.Error(), "names no API") {
		t.Errorf("empty apis: %v", err)
	}
}

func TestAVSRProfileSurvivesCanonicalExport(t *testing.T) {
	cfg, err := loadListenerAPIs(t, "      chat_completions: {}\n      systemone: {vsr: {redact: false, router_turns: true, router_goal: save}}", "chat_completions")
	if err != nil {
		t.Fatal(err)
	}
	profile := cfg.Listeners[0].APIs["systemone"].VSR
	if profile == nil || *profile.Redact || !*profile.RouterTurns || profile.RouterGoal != "save" || profile.Guard != nil {
		t.Fatalf("profile %+v", profile)
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("%v\n%s", err, exported)
	}
	if again := reloaded.Listeners[0].APIs["systemone"].VSR; again == nil || *again.Redact || again.RouterGoal != "save" || again.Guard != nil {
		t.Fatalf("exported profile %+v\n%s", again, exported)
	}
}

func TestAListenerWithoutAPIsServesTheLLMAPIs(t *testing.T) {
	listener := Listener{Name: "plain"}
	if got := strings.Join(listener.ServedAPIs(), ","); got != "chat_completions,responses,messages,images,videos" {
		t.Fatalf("served %s", got)
	}
	if models, restricted := listener.LLMModels(); models != nil || restricted {
		t.Fatalf("models %v restricted %v", models, restricted)
	}
	systemOne := Listener{APIs: ListenerAPIs{"systemone": {}}}
	if models, restricted := systemOne.LLMModels(); len(models) != 0 || !restricted {
		t.Fatalf("a systemone-only listener accepts LLM models %v", models)
	}
	if schema := systemOne.APIs["systemone"].SchemaID("systemone"); schema != "vllm-sr/5.0" {
		t.Fatalf("listener schema %s", schema)
	}
	if limit := systemOne.APIs["systemone"].MaxDeadline(); limit.Minutes() != 10 {
		t.Fatalf("default max deadline %s", limit)
	}
}

func TestMigrateMovesListenerModelsAndSystemOneIntoAPIs(t *testing.T) {
	before := `version: v0.5
listeners:
  - name: public
    port: 8899
    models: [m]
    systemone:
      models: [decision-public]
      max_deadline_ms: 120000
  - name: chat
    port: 8900
    models: [m]
providers:
  defaults:
    model: m
  models:
    - name: m
      api_format: openai
      backend_refs:
        - endpoint: 127.0.0.1:8000
          protocol: http
          provider: vllm
global:
  model_catalog:
    deployments:
      decision:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-0.3B
        public_name: decision-public
recipes:
  - name: default
`
	if _, err := ParseYAMLBytes([]byte(before)); err == nil || !strings.Contains(err.Error(), "listeners[public].models: removed_field") ||
		!strings.Contains(err.Error(), "listeners[public].systemone: removed_field") {
		t.Fatalf("the earlier listener fields load: %v", err)
	}
	migrated, _, err := MigrateConfig([]byte(before))
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseYAMLBytes(migrated)
	if err != nil {
		t.Fatalf("%v\n%s", err, migrated)
	}
	public := cfg.Listeners[0]
	if got := strings.Join(public.ServedAPIs(), ","); got != "chat_completions,responses,messages,images,videos,systemone" {
		t.Fatalf("public serves %s\n%s", got, migrated)
	}
	for _, name := range []string{"chat_completions", "responses", "messages", "images", "videos"} {
		if api := public.APIs[name]; strings.Join(api.Models, ",") != "m" {
			t.Fatalf("%s models %v", name, api.Models)
		}
	}
	systemOne := public.APIs["systemone"]
	if systemOne.Schema != "vllm-sr/4.0" || strings.Join(systemOne.Models, ",") != "decision-public" || systemOne.MaxDeadline().Minutes() != 2 {
		t.Fatalf("systemone %+v", systemOne)
	}
	if chat := cfg.Listeners[1]; chat.Serves("systemone") || strings.Join(chat.APIs["messages"].Models, ",") != "m" {
		t.Fatalf("chat %+v", chat.APIs)
	}
	if strings.Contains(string(migrated), "api_format: chat_completions") {
		t.Fatal("migrate rewrote an api_format alias")
	}
}
