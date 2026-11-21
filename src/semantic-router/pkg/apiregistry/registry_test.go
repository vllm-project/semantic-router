package apiregistry

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestEveryAPIHasSchemasAndDefaults(t *testing.T) {
	for _, api := range All() {
		if len(api.Schemas) == 0 || len(api.Paths) == 0 {
			t.Fatalf("%s has no schemas or paths", api.Name)
		}
		for _, id := range []string{api.ListenerDefault, api.UpstreamDefault} {
			if _, ok := api.Schema(id); !ok {
				t.Fatalf("%s's default %q is not one of its schemas", api.Name, id)
			}
		}
		if api.Versioned != (len(api.Schemas) > 1) {
			t.Fatalf("%s is versioned=%v with %d schemas", api.Name, api.Versioned, len(api.Schemas))
		}
	}
}

func TestTheLLMAPIsKeepTheirCodecs(t *testing.T) {
	want := map[Name]llmprotocol.WireFormat{
		ChatCompletions: llmprotocol.OpenAIChatV1,
		Responses:       llmprotocol.OpenAIResponsesV1,
		Messages:        llmprotocol.AnthropicMessagesV1,
		Images:          llmprotocol.OpenAIImagesV1,
	}
	for name, format := range want {
		api, err := Resolve(string(name))
		if err != nil {
			t.Fatal(err)
		}
		schema, _ := api.UpstreamSchema("")
		if schema.WireFormat != format {
			t.Fatalf("%s speaks %q, want %q", name, schema.WireFormat, format)
		}
	}
}

func TestSystemOneDefaultsDifferBySide(t *testing.T) {
	api, err := Resolve("systemone")
	if err != nil {
		t.Fatal(err)
	}
	listener, _ := api.ListenerSchema("")
	upstream, _ := api.UpstreamSchema("")
	if listener.ID != SchemaVLLMSR50 || upstream.ID != SchemaSystemOne02 {
		t.Fatalf("defaults listener %s, upstream %s", listener.ID, upstream.ID)
	}
	if _, err := api.ListenerSchema(SchemaVLLMSR40); err != nil {
		t.Fatal(err)
	}
	if _, err := api.UpstreamSchema("vllm-sr/9.9"); err == nil || !strings.Contains(err.Error(), "vllm-sr/5.0, vllm-sr/4.0, systemone/0.2") {
		t.Fatalf("an unknown schema: %v", err)
	}
}

func TestAnUnversionedAPITakesNoListenerSchema(t *testing.T) {
	api, _ := Resolve("chat_completions")
	if _, err := api.ListenerSchema(SchemaOpenAIV1); err == nil {
		t.Fatal("a listener named a schema for chat_completions")
	}
}

func TestTheVendorNamesAreAliases(t *testing.T) {
	for alias, want := range map[string]Name{"openai": ChatCompletions, "anthropic": Messages} {
		api, err := Resolve(alias)
		if err != nil || api.Name != want || Canonical(alias) != string(want) {
			t.Fatalf("%s resolves to %q (%v), want %q", alias, api.Name, err, want)
		}
	}
	if Canonical("responses") != "responses" {
		t.Fatal("a canonical name is its own canonical name")
	}
	if _, err := Resolve("realtime"); err == nil || !strings.Contains(err.Error(), "chat_completions, responses, messages, images, videos, systemone") {
		t.Fatalf("an unknown api: %v", err)
	}
}

func TestPathsBelongToOneAPI(t *testing.T) {
	for path, want := range map[string]Name{
		"/v1/chat/completions":       ChatCompletions,
		"/v1/responses/resp_1":       Responses,
		"/v1/messages/count_tokens":  Messages,
		"/v1/images/generations":     Images,
		"/v1/videos":                 Videos,
		"/v1/videos/video-route:job": Videos,
		"/v1/systemone/batch":        SystemOne,
		"/v1/decisions":              SystemOne,
	} {
		api, ok := ForPath(path)
		if !ok || api.Name != want {
			t.Fatalf("%s belongs to %q, want %q", path, api.Name, want)
		}
	}
}

func TestPathsIncludeTheAzureForms(t *testing.T) {
	for path, want := range map[string]Name{
		"/v1/chat/completions":                              ChatCompletions,
		"/openai/deployments/vllm-sr/auto/chat/completions": ChatCompletions,
		"/openai/v1/chat/completions?api-version=preview":   ChatCompletions,
		"/v1/responses/resp_1/input_items":                  Responses,
		"/openai/v1/responses":                              Responses,
		"/v1/messages/count_tokens":                         Messages,
		"/v1/images/generations/":                           Images,
		"/v1/videos/video-route:job":                        Videos,
		"/v1/systemone/batch":                               SystemOne,
		"/v1/decisions":                                     SystemOne,
	} {
		api, ok := ForPath(path)
		if !ok || api.Name != want {
			t.Fatalf("%s belongs to %q, want %q", path, api.Name, want)
		}
	}
	for _, path := range []string{"/v1/models", "/v1/completions", "/v1/systemone/other", "/health"} {
		if api, ok := ForPath(path); ok {
			t.Fatalf("%s belongs to no API, got %s", path, api.Name)
		}
	}
}

func TestAListenerServesSystemOneInItsOwnSchemas(t *testing.T) {
	api, _ := Resolve("systemone")
	if _, err := api.ListenerSchema(SchemaVLLMSR40); err != nil {
		t.Fatal(err)
	}
	if _, err := api.ListenerSchema(SchemaSystemOne02); err == nil || !strings.Contains(err.Error(), "vllm-sr/5.0 or vllm-sr/4.0") {
		t.Fatalf("a listener serving systemone/0.2: %v", err)
	}
	if _, err := api.UpstreamSchema(SchemaSystemOne02); err != nil {
		t.Fatal(err)
	}
}
