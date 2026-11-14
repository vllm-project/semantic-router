// Package apiregistry is the one registry of the APIs the Router speaks:
// served by a listener to its clients, and called on an upstream by a
// provider model or a deployment. An API is named by the API itself, not by
// its vendor, and declares its paths, the kind of model it serves, its schema
// IDs with the default on each side, its own settings, and the codec of each
// schema ID. A schema ID names whose specification and which version
// (openai/v1, anthropic/2023-06-01, systemone/0.2, vllm-sr/5.0).
//
// Listeners and upstreams use the same names and IDs. A new API, or a new
// version of one, is one entry here and its codec; neither changes the
// configuration's shape or any caller.
package apiregistry

import (
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemonecodec"
)

// Name names an API, as listeners[].apis keys and providers.models[].api_format
// values write it.
type Name string

const (
	ChatCompletions Name = "chat_completions"
	Responses       Name = "responses"
	Messages        Name = "messages"
	Images          Name = "images"
	SystemOne       Name = "systemone"
)

// ModelKind is the kind of model an API serves.
type ModelKind string

const (
	KindLLM      ModelKind = "llm"
	KindDecision ModelKind = "decision"
)

// The schema IDs of the APIs.
const (
	SchemaOpenAIV1          = "openai/v1"
	SchemaAnthropic20230601 = "anthropic/2023-06-01"
	SchemaSystemOne02       = string(systemonecodec.SchemaSystemOne02)
	SchemaVLLMSR50          = string(systemonecodec.SchemaVLLMSR50)
	SchemaVLLMSR40          = "vllm-sr/4.0"
)

// Setting is a setting an API takes on a listener besides schema and models.
type Setting struct {
	Name    string
	Summary string
}

// Schema is one version of an API and the codec that speaks it: a wire
// format of the LLM protocol codecs, or a System One schema codec.
type Schema struct {
	ID         string
	WireFormat llmprotocol.WireFormat
	SystemOne  systemonecodec.SchemaID
}

// API is one registry entry.
type API struct {
	Name    Name
	Summary string
	// Paths are the API's documented request paths; Pattern, an anchored
	// RE2 expression, matches every path the API answers, the Azure OpenAI
	// forms among them. Envoy routes by it as the Router gates by it.
	Paths   []string
	Pattern string
	Kind    ModelKind
	Schemas []Schema
	// ListenerSchemas, when set, are the only schemas a listener may serve;
	// the others are spoken to upstreams only.
	ListenerSchemas []string
	// ListenerDefault and UpstreamDefault are the schema a listener serves
	// and an upstream speaks when the configuration names none.
	ListenerDefault string
	UpstreamDefault string
	// Versioned says a listener may choose the schema it serves; an API with
	// one schema takes none.
	Versioned bool
	// DefaultOn says a listener without apis serves the API.
	DefaultOn bool
	Settings  []Setting
	// Aliases are other names of the API, accepted wherever an API is named
	// and read as its name: the vendor names earlier configurations used.
	Aliases []string
}

var registry = []API{
	{
		Name: ChatCompletions, Summary: "OpenAI Chat Completions", Paths: []string{"/v1/chat/completions"}, Kind: KindLLM,
		Pattern:         `^(/v1/chat/completions|/openai/v1/chat/completions|/openai/deployments/.+/chat/completions)$`,
		Schemas:         []Schema{{ID: SchemaOpenAIV1, WireFormat: llmprotocol.OpenAIChatV1}},
		ListenerDefault: SchemaOpenAIV1, UpstreamDefault: SchemaOpenAIV1, DefaultOn: true, Aliases: []string{"openai"},
	},
	{
		Name: Responses, Summary: "OpenAI Responses", Paths: []string{"/v1/responses"}, Kind: KindLLM,
		Pattern:         `^(/v1/responses(/[^/]+(/input_items)?)?|/openai/responses|/openai/v1/responses)$`,
		Schemas:         []Schema{{ID: SchemaOpenAIV1, WireFormat: llmprotocol.OpenAIResponsesV1}},
		ListenerDefault: SchemaOpenAIV1, UpstreamDefault: SchemaOpenAIV1, DefaultOn: true,
	},
	{
		Name: Messages, Summary: "Anthropic Messages", Paths: []string{"/v1/messages", "/v1/messages/count_tokens"}, Kind: KindLLM,
		Pattern:         `^/v1/messages(/count_tokens)?$`,
		Schemas:         []Schema{{ID: SchemaAnthropic20230601, WireFormat: llmprotocol.AnthropicMessagesV1}},
		ListenerDefault: SchemaAnthropic20230601, UpstreamDefault: SchemaAnthropic20230601, DefaultOn: true, Aliases: []string{"anthropic"},
	},
	{
		Name: Images, Summary: "OpenAI Images", Paths: []string{"/v1/images/generations"}, Kind: KindLLM,
		Pattern:         `^/v1/images/generations$`,
		Schemas:         []Schema{{ID: SchemaOpenAIV1, WireFormat: llmprotocol.OpenAIImagesV1}},
		ListenerDefault: SchemaOpenAIV1, UpstreamDefault: SchemaOpenAIV1, DefaultOn: true,
	},
	{
		Name: SystemOne, Summary: "System One decisions", Kind: KindDecision,
		Paths:   []string{"/v1/systemone", "/v1/systemone/batch", "/v1/decisions", "/v1/systemone/models"},
		Pattern: `^/v1/(systemone(/batch|/models)?|decisions)$`,
		Schemas: []Schema{
			{ID: SchemaVLLMSR50, SystemOne: systemonecodec.SchemaVLLMSR50},
			{ID: SchemaVLLMSR40, SystemOne: systemonecodec.SchemaID(SchemaVLLMSR40)},
			{ID: SchemaSystemOne02, SystemOne: systemonecodec.SchemaSystemOne02},
		},
		ListenerDefault: SchemaVLLMSR50, UpstreamDefault: SchemaSystemOne02, Versioned: true,
		ListenerSchemas: []string{SchemaVLLMSR50, SchemaVLLMSR40},
		Settings: []Setting{
			{Name: "max_deadline_ms", Summary: "the most a request's deadline may ask a directly called model to take"},
			{Name: "vsr", Summary: "what the vSR agent plugins pointed at the listener turn on, from its System One listing"},
		},
	},
}

// All are the registry's APIs, in order.
func All() []API {
	apis := make([]API, len(registry))
	for index, api := range registry {
		apis[index] = api.clone()
	}
	return apis
}

// Names are the registry's API names, in order.
func Names() []string {
	names := make([]string, len(registry))
	for index, api := range registry {
		names[index] = string(api.Name)
	}
	return names
}

// AcceptedNames are every name an API may be given by: the registry's names
// in order, then their aliases.
func AcceptedNames() []string {
	names := Names()
	for _, api := range registry {
		names = append(names, api.Aliases...)
	}
	return names
}

// SchemaIDs are the schema IDs of every API, each once, in registry order.
func SchemaIDs() []string {
	var ids []string
	for _, api := range registry {
		for _, schema := range api.Schemas {
			if !slices.Contains(ids, schema.ID) {
				ids = append(ids, schema.ID)
			}
		}
	}
	return ids
}

// Lookup is the API a name or one of its aliases names.
func Lookup(name string) (API, bool) {
	for _, entry := range registry {
		if string(entry.Name) == name || slices.Contains(entry.Aliases, name) {
			return entry.clone(), true
		}
	}
	return API{}, false
}

// Resolve is the API a name or alias names, or an error listing the names.
func Resolve(name string) (API, error) {
	api, ok := Lookup(name)
	if !ok {
		return API{}, fmt.Errorf("api %q is not supported; the apis are %s", name, strings.Join(Names(), ", "))
	}
	return api, nil
}

// Canonical is the name of the API a name or alias names, or the name itself
// when the registry has no such API.
func Canonical(name string) string {
	if api, ok := Lookup(name); ok {
		return string(api.Name)
	}
	return name
}

// Schema is the API's schema of an ID.
func (a API) Schema(id string) (Schema, bool) {
	for _, schema := range a.Schemas {
		if schema.ID == id {
			return schema, true
		}
	}
	return Schema{}, false
}

// SchemaIDs are the API's schema IDs, in order.
func (a API) SchemaIDs() []string {
	ids := make([]string, len(a.Schemas))
	for index, schema := range a.Schemas {
		ids[index] = schema.ID
	}
	return ids
}

// ListenerSchema is the schema a listener serves the API in: id, else the
// listener default. A listener names a schema only for a versioned API.
func (a API) ListenerSchema(id string) (Schema, error) {
	if id == "" {
		schema, _ := a.Schema(a.ListenerDefault)
		return schema, nil
	}
	if !a.Versioned {
		return Schema{}, fmt.Errorf("api %s has one schema, %s, so a listener names none", a.Name, a.ListenerDefault)
	}
	if len(a.ListenerSchemas) > 0 && !slices.Contains(a.ListenerSchemas, id) {
		return Schema{}, fmt.Errorf("a listener serves api %s in %s, not %q", a.Name, strings.Join(a.ListenerSchemas, " or "), id)
	}
	return a.resolveSchema(id)
}

// UpstreamSchema is the schema an upstream speaks the API in: id, else the
// upstream default.
func (a API) UpstreamSchema(id string) (Schema, error) {
	if id == "" {
		schema, _ := a.Schema(a.UpstreamDefault)
		return schema, nil
	}
	return a.resolveSchema(id)
}

func (a API) resolveSchema(id string) (Schema, error) {
	schema, ok := a.Schema(id)
	if !ok {
		return Schema{}, fmt.Errorf("schema %q is not a schema of api %s; its schemas are %s", id, a.Name, strings.Join(a.SchemaIDs(), ", "))
	}
	return schema, nil
}

// Setting is the API's setting of a name.
func (a API) Setting(name string) (Setting, bool) {
	for _, setting := range a.Settings {
		if setting.Name == name {
			return setting, true
		}
	}
	return Setting{}, false
}

// ModelsPath is the model listing every listener serves: it lists the models
// of the APIs the listener serves.
const ModelsPath = "/v1/models"

var patterns = func() map[Name]*regexp.Regexp {
	compiled := make(map[Name]*regexp.Regexp, len(registry))
	for _, api := range registry {
		compiled[api.Name] = regexp.MustCompile(api.Pattern)
	}
	return compiled
}()

// Serves reports whether the API answers a request path, without its query
// and a trailing slash.
func (a API) Serves(path string) bool {
	pattern, ok := patterns[a.Name]
	return ok && pattern.MatchString(normalizePath(path))
}

func normalizePath(path string) string {
	if index := strings.IndexByte(path, '?'); index >= 0 {
		path = path[:index]
	}
	if len(path) > 1 {
		path = strings.TrimSuffix(path, "/")
	}
	return path
}

// ForPath is the API a request path belongs to.
func ForPath(path string) (API, bool) {
	for _, api := range registry {
		if api.Serves(path) {
			return api.clone(), true
		}
	}
	return API{}, false
}

func (a API) clone() API {
	a.Paths = slices.Clone(a.Paths)
	a.Schemas = slices.Clone(a.Schemas)
	a.ListenerSchemas = slices.Clone(a.ListenerSchemas)
	a.Settings = slices.Clone(a.Settings)
	a.Aliases = slices.Clone(a.Aliases)
	return a
}
