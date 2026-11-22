package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiregistry"
)

func TestUnservedModelsRouteToAnEntrypointTheAPIAccepts(t *testing.T) {
	cfg, err := loadListenerAPIs(t, "      chat_completions: {models: [vllm-sr/auto], unserved_models: {policy: route, route_to: vllm-sr/auto}}\n"+
		"      messages: {unserved_models: {policy: reject}}\n      responses: {}", "openai")
	if err != nil {
		t.Fatal(err)
	}
	listener := cfg.Listeners[0]
	for api, want := range map[string]string{"chat_completions": "vllm-sr/auto", "messages": "", "responses": ""} {
		entry, _ := listener.API(apiregistry.Name(api))
		if got := entry.UnservedRoute(); got != want {
			t.Fatalf("%s routes unserved models as %q, want %q", api, got, want)
		}
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(exported), "route_to: vllm-sr/auto") {
		t.Fatalf("export lost the route:\n%s", exported)
	}
}

func TestUnservedModelsAreChecked(t *testing.T) {
	for apis, want := range map[string]string{
		"      chat_completions: {unserved_models: {policy: route}}":                                      "apis.chat_completions.unserved_models.route_to: invalid_value: is required with policy route",
		"      chat_completions: {unserved_models: {policy: reject, route_to: vllm-sr/auto}}":             "apis.chat_completions.unserved_models.route_to: invalid_value: goes with policy route only",
		"      chat_completions: {unserved_models: {route_to: vllm-sr/auto}}":                             "apis.chat_completions.unserved_models.route_to: invalid_value: goes with policy route only",
		"      chat_completions: {unserved_models: {policy: forward}}":                                    `apis.chat_completions.unserved_models.policy: invalid_value: "forward" is not a policy; use reject or route`,
		"      chat_completions: {models: [vllm-sr/auto], unserved_models: {policy: route, route_to: m}}": `apis.chat_completions.unserved_models.route_to: invalid_value: "m" is not one of the API's models`,
		"      chat_completions: {unserved_models: {policy: route, route_to: m}}":                         `apis.chat_completions.unserved_models.route_to: invalid_value: "m" names no entrypoint`,
		"      images: {unserved_models: {policy: route, route_to: vllm-sr/auto}}":                        "apis.images.unserved_models: invalid_value: the images API takes no unserved_models setting",
		"      systemone: {unserved_models: {policy: route, route_to: vllm-sr/auto}}\n      messages: {}": "apis.systemone.unserved_models: invalid_value: the systemone API takes no unserved_models setting",
	} {
		_, err := loadListenerAPIs(t, apis, "openai")
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("%s:\ngot  %v\nwant %s", apis, err, want)
		}
	}
}
