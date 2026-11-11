package config

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiregistry"
)

// CodeDuplicateAPI is the load error of an apis map that names one API twice,
// by its name and by an alias.
const CodeDuplicateAPI = "duplicate_api"

// ListenerAPIs are the APIs a listener serves, by API name.
type ListenerAPIs map[string]ListenerAPI

// ListenerAPI is one API a listener serves. Every entry has the same shape;
// which settings an API takes is the API registry's.
type ListenerAPI struct {
	// Schema is the version of a versioned API (systemone) the listener
	// speaks to its clients; omitted, the API's listener default. Naming one
	// on an API without versions is a load error.
	Schema string `yaml:"schema,omitempty" json:"schema,omitempty" jsonschema:"enum=vllm-sr/5.0,enum=vllm-sr/4.0"`
	// Models, when set, are the only request models the API accepts, by
	// exact `model` value; others get 403 model_not_allowed. Omitted, the API
	// accepts every model of its kind: the LLM models for the LLM APIs, the
	// public System One models for systemone.
	Models []string `yaml:"models,omitempty" json:"models,omitempty"`
	// MaxDeadlineMs (systemone) bounds how long the listener waits for a
	// model it calls directly, however long a request's options.deadline_ms
	// asks for. Omitted, DefaultSystemOneMaxDeadlineMs.
	MaxDeadlineMs *int `yaml:"max_deadline_ms,omitempty" json:"max_deadline_ms,omitempty" jsonschema:"minimum=1"`
}

// settingValues are the settings an entry sets, by name, for the registry's
// check of which settings the API takes.
func (a ListenerAPI) settingValues() []string {
	var set []string
	if a.MaxDeadlineMs != nil {
		set = append(set, "max_deadline_ms")
	}
	return set
}

// EffectiveAPIs are the APIs the listener serves: its apis, or without them
// the APIs on by default (chat_completions, responses, messages and images),
// each with no settings.
func (l Listener) EffectiveAPIs() ListenerAPIs {
	if l.APIs != nil {
		return l.APIs
	}
	apis := ListenerAPIs{}
	for _, api := range apiregistry.All() {
		if api.DefaultOn {
			apis[string(api.Name)] = ListenerAPI{}
		}
	}
	return apis
}

// API is the listener's entry for an API, and whether it serves the API.
func (l Listener) API(name apiregistry.Name) (ListenerAPI, bool) {
	api, ok := l.EffectiveAPIs()[string(name)]
	return api, ok
}

// Serves reports whether the listener serves an API.
func (l Listener) Serves(name apiregistry.Name) bool {
	_, ok := l.API(name)
	return ok
}

// ServedAPIs are the names of the APIs the listener serves, in registry
// order.
func (l Listener) ServedAPIs() []string {
	apis := l.EffectiveAPIs()
	var names []string
	for _, api := range apiregistry.All() {
		if _, ok := apis[string(api.Name)]; ok {
			names = append(names, string(api.Name))
		}
	}
	return names
}

// LLMModels are the request models the listener's LLM APIs accept, and
// whether they are restricted: nil and false when one of them accepts every
// model. A listener that serves no LLM API accepts none.
func (l Listener) LLMModels() ([]string, bool) {
	var models []string
	served := false
	for name, api := range l.EffectiveAPIs() {
		entry, ok := apiregistry.Lookup(name)
		if !ok || entry.Kind != apiregistry.KindLLM {
			continue
		}
		served = true
		if api.Models == nil {
			return nil, false
		}
		for _, model := range api.Models {
			if !slices.Contains(models, model) {
				models = append(models, model)
			}
		}
	}
	if !served {
		return []string{}, true
	}
	sort.Strings(models)
	return models, true
}

// Allows reports whether the API accepts a request model.
func (a ListenerAPI) Allows(model string) bool {
	return a.Models == nil || slices.Contains(a.Models, model)
}

// MaxDeadline is the longest the listener's System One API waits for a model
// it calls directly.
func (a ListenerAPI) MaxDeadline() time.Duration {
	if a.MaxDeadlineMs == nil {
		return DefaultSystemOneMaxDeadlineMs * time.Millisecond
	}
	return time.Duration(*a.MaxDeadlineMs) * time.Millisecond
}

// SchemaID is the schema the listener speaks on an API: its schema, or the
// API's listener default.
func (a ListenerAPI) SchemaID(name apiregistry.Name) string {
	if a.Schema != "" {
		return a.Schema
	}
	api, ok := apiregistry.Lookup(string(name))
	if !ok {
		return ""
	}
	return api.ListenerDefault
}

// SystemOneModels are the public System One models a listener's systemone
// API serves: its models, or without them every public System One model of
// the configuration. A listener that does not serve systemone serves none.
func (c *RouterConfig) SystemOneModels(listener Listener) []string {
	api, ok := listener.API(apiregistry.SystemOne)
	if !ok {
		return nil
	}
	if api.Models != nil {
		return api.Models
	}
	return c.PublicSystemOneModels()
}

// PublicSystemOneModels are every model name a System One request can
// name: the routed System One entrypoints, the providers.models on
// api_format systemone, and the managed deployments with a public name.
func (c *RouterConfig) PublicSystemOneModels() []string {
	if c == nil {
		return nil
	}
	seen := map[string]bool{}
	var models []string
	add := func(name string) {
		if name != "" && !seen[name] {
			seen[name] = true
			models = append(models, name)
		}
	}
	for _, entrypoint := range c.EffectiveEntrypoints(SystemOneAPI) {
		for _, name := range normalizeEntrypointNames(entrypoint.ModelNames) {
			add(name)
		}
	}
	for name, backend := range c.ModelConfig {
		if backend.APIFormat == APIFormatSystemOne {
			add(name)
		}
	}
	for _, deployment := range c.ModelDeployments {
		if deployment.Provider == ModelRuntimeProvider {
			name := deployment.PublicModelName()
			if _, _, err := c.ResolveSystemOneDeployment(name); err == nil {
				add(name)
			}
		}
	}
	sort.Strings(models)
	return models
}

// normalizeListenerAPIs reads every apis key as the API it names, an alias
// as its API, and refuses a map that names one API twice.
func normalizeListenerAPIs(listeners []Listener) error {
	for index := range listeners {
		listener := &listeners[index]
		if listener.APIs == nil {
			continue
		}
		normalized := make(ListenerAPIs, len(listener.APIs))
		written := make(map[string]string, len(listener.APIs))
		keys := make([]string, 0, len(listener.APIs))
		for key := range listener.APIs {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			name := apiregistry.Canonical(key)
			if earlier, ok := written[name]; ok {
				first, second := earlier, key
				if first == name {
					first, second = second, first
				}
				return CheckError(listenerPath(*listener, "apis"), CodeDuplicateAPI,
					fmt.Sprintf("%s and %s name the same API, %s; keep one (%s is an alias of %s)", first, second, name, first, name))
			}
			written[name] = key
			normalized[name] = listener.APIs[key]
		}
		listener.APIs = normalized
	}
	return nil
}

func listenerPath(listener Listener, rest string) string {
	label := listener.Name
	if label == "" {
		label = "0"
	}
	return "listeners[" + label + "]." + rest
}

// validateListenerAPIs checks each API a listener names against the API
// registry: the API exists, a schema only on a versioned API and one of its
// schemas, the settings the API takes, and exact, distinct model names.
func validateListenerAPIs(cfg *RouterConfig, listener Listener) error {
	if listener.APIs != nil && len(listener.APIs) == 0 {
		return CheckError(listenerPath(listener, "apis"), CodeInvalidValue,
			"names no API; name the APIs the listener serves, or omit apis to serve chat_completions, responses, messages and images")
	}
	names := make([]string, 0, len(listener.APIs))
	for name := range listener.APIs {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		entry := listener.APIs[name]
		path := listenerPath(listener, "apis."+name)
		api, err := apiregistry.Resolve(name)
		if err != nil {
			return CheckError(path, CodeInvalidValue, err.Error())
		}
		if _, err := api.ListenerSchema(entry.Schema); err != nil {
			return CheckError(path+".schema", CodeInvalidValue, err.Error())
		}
		for _, setting := range entry.settingValues() {
			if _, ok := api.Setting(setting); !ok {
				return CheckError(path+"."+setting, CodeInvalidValue,
					fmt.Sprintf("the %s API takes no %s setting", api.Name, setting))
			}
		}
		if limit := entry.MaxDeadlineMs; limit != nil && *limit <= 0 {
			return CheckError(path+".max_deadline_ms", CodeInvalidValue,
				fmt.Sprintf("must be greater than zero, got %d", *limit))
		}
		if err := validateListenerAPIModels(path, entry.Models); err != nil {
			return err
		}
		if api.Name == apiregistry.SystemOne {
			if err := validatePublishedSystemOneModels(cfg, path, entry.Models); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateListenerAPIModels(path string, models []string) error {
	if models != nil && len(models) == 0 {
		return CheckError(path+".models", CodeInvalidValue, "lists no model; omit it to accept every model of the API")
	}
	seen := make(map[string]bool, len(models))
	for index, model := range models {
		if model == "" || strings.TrimSpace(model) != model {
			return CheckError(fmt.Sprintf("%s.models[%d]", path, index), CodeInvalidValue,
				fmt.Sprintf("%q must be a model name without surrounding spaces", model))
		}
		if seen[model] {
			return CheckError(path+".models", CodeInvalidValue, fmt.Sprintf("lists %q twice", model))
		}
		seen[model] = true
	}
	return nil
}

// validatePublishedSystemOneModels checks that every model a systemone API
// names is one a System One request can reach.
func validatePublishedSystemOneModels(cfg *RouterConfig, path string, models []string) error {
	for _, model := range models {
		// A routed entrypoint is not one deployment. A provider alias is
		// checked in validateNativeRoutingConfig, published or not.
		if _, routed := cfg.ResolveEntrypoint(SystemOneAPI, model); routed || cfg.IsSystemOneBackend(model) {
			continue
		}
		name, deployment, err := cfg.ResolveSystemOneDeployment(model)
		if err != nil {
			return CheckError(path+".models", CodeInvalidValue, err.Error())
		}
		if err := deployment.ValidateDecisionInput(name); err != nil {
			return CheckError(path+".models", CodeInvalidValue, err.Error())
		}
	}
	return nil
}
