package config

import (
	"fmt"
	"maps"
	"net"
	"slices"
	"strings"
)

// A provider backend whose auth is credential: caller takes the provider
// login the client itself sent the Router, as Claude Code sends its Claude
// login and Codex its ChatGPT login, so the gateway routes a person's own
// subscription without holding a key. Only the client's own request presents it: a call the Router makes
// on its own behalf, an in-process hop and a response cache have no login of
// that person's, so a decision that would do so with a caller model fails to
// load. Caller models may sit beside models with a key of their own or none,
// such as a local model for easy turns. One model keeps one payer, since its
// backends share one credential source, and every caller backend names one
// provider.

// CallerCredentialHosts are the providers a caller backend may name, each
// with the hosts that serve the logins it issues: Anthropic's API, OpenAI's
// API for a key and ChatGPT's Codex backend for a ChatGPT login.
var CallerCredentialHosts = map[string][]string{
	"anthropic": {"api.anthropic.com"},
	"openai":    {"api.openai.com", "chatgpt.com"},
}

// validateCallerUpstreams checks every caller backend against what the
// rest of the configuration would do with a person's login.
func validateCallerUpstreams(cfg *RouterConfig) error {
	first, ok := cfg.firstCallerBackend()
	if !ok {
		return nil
	}
	provider := ""
	for _, model := range sortedModelKeys(cfg.ModelConfig) {
		for index, endpoint := range cfg.ModelConfig[model].PreferredEndpoints {
			path := fmt.Sprintf("providers.models[%s].backend_refs[%d]", model, index)
			profile, err := cfg.GetProviderProfileForEndpoint(endpoint)
			if err != nil || !profile.Caller() {
				continue
			}
			if err := validateCallerBackend(cfg, path, endpoint, profile); err != nil {
				return err
			}
			if provider == "" {
				provider = profile.Type
			} else if !strings.EqualFold(provider, profile.Type) {
				return fmt.Errorf("%s: names provider %q while %s names %q; a login reaches only the provider that issued it, so every caller backend names one provider", path, profile.Type, first, provider)
			}
		}
	}
	for index, listener := range cfg.Listeners {
		if len(listener.APIKeys) > 0 {
			return fmt.Errorf("listeners[%d]: api_keys authenticate clients with the Authorization header, which carries the caller's login to %s; serve caller backends from a listener without api_keys", index, first)
		}
	}
	return cfg.validateCallerRequestOnly(first)
}

// validateCallerBackend keeps the login on the API that issued it, over TLS,
// or on an endpoint on this machine.
func validateCallerBackend(cfg *RouterConfig, path, endpointName string, profile *ProviderProfile) error {
	hosts, ok := CallerCredentialHosts[strings.ToLower(strings.TrimSpace(profile.Type))]
	if !ok {
		return fmt.Errorf("%s: auth.credential: caller sends the client's login, which only %s issue; provider %q would receive a login it did not issue", path, strings.Join(slices.Sorted(maps.Keys(CallerCredentialHosts)), ", "), profile.Type)
	}
	for _, endpoint := range cfg.VLLMEndpoints {
		if endpoint.Name != endpointName {
			continue
		}
		if isLoopbackAddress(endpoint.Address) {
			continue
		}
		host := strings.ToLower(strings.TrimSuffix(endpoint.Address, "."))
		if !strings.EqualFold(endpoint.Protocol, "https") || !slices.Contains(hosts, host) {
			return fmt.Errorf("%s: auth.credential: caller sends the client's login, so the backend is https://%s, or an endpoint on this machine", path, strings.Join(hosts, " or https://"))
		}
	}
	return nil
}

// validateCallerRequestOnly refuses whatever would call a caller model
// without the client's own request, an in-process hop and an answer kept in
// a response cache for other requests, and a system prompt the Router writes
// over the client's, which both providers refuse on a person's login. A
// decision that reaches no caller model may do all of these.
func (c *RouterConfig) validateCallerRequestOnly(first string) error {
	recipes := c.Recipes
	if len(recipes) == 0 {
		recipes = []RoutingRecipe{*c.DefaultRecipe()}
	}
	if ResponseCachePluginOf(c.GlobalPlugins) != nil {
		return fmt.Errorf("global.plugins: response_cache keeps answers for other requests, and %s takes the caller's login, which names no user to keep them for; remove response_cache", first)
	}
	for _, recipe := range recipes {
		// A request may name a caller model itself, past every decision.
		if ResponseCachePluginOf(recipe.Profile.DefaultPlugins) != nil {
			return fmt.Errorf("recipes[%s].plugins: response_cache keeps answers for other requests, and %s takes the caller's login; remove response_cache", recipe.Name, first)
		}
		for _, decision := range recipe.Profile.Decisions {
			if err := c.validateCallerDecision(fmt.Sprintf("recipes[%s].decisions[%s]", recipe.Name, decision.Name), decision); err != nil {
				return err
			}
		}
	}
	return nil
}

// validateCallerDecision refuses an in-process call to a caller model on any
// decision, and a response cache or system prompt on a decision whose
// request can reach one.
func (c *RouterConfig) validateCallerDecision(path string, decision Decision) error {
	for _, called := range decisionCalledModels(decision) {
		if c.isCallerModel(called) {
			return fmt.Errorf("%s: the Router calls %q in process, and a hop never carries the caller's login that it takes; route this decision to one model", path, called)
		}
	}
	caller, ok := c.decisionCallerModel(decision)
	if !ok {
		return nil
	}
	if ResponseCachePluginOf(decision.DeclaredPlugins()) != nil {
		return fmt.Errorf("%s.plugins: response_cache keeps answers for other requests, and %s takes the caller's login; remove response_cache", path, caller)
	}
	if prompt := decision.GetSystemPromptConfig(); prompt.IsEnabled() {
		return fmt.Errorf("%s.plugins: system_prompt (mode %s) puts a system message before the client's own, which the provider refuses on a person's login (Anthropic answers 429, ChatGPT's Codex backend 400); %s takes the caller's login, remove system_prompt", path, prompt.EffectiveMode(), caller)
	}
	return nil
}

// decisionCallerModel returns a caller model the decision can send the
// client's request to: one of its modelRefs, or the default model when it
// names none.
func (c *RouterConfig) decisionCallerModel(decision Decision) (string, bool) {
	models := make([]string, 0, len(decision.ModelRefs)+1)
	for _, ref := range decision.ModelRefs {
		models = append(models, ref.Model)
	}
	if len(models) == 0 {
		if c.DefaultModel == "" {
			// Without a default model the route is not known here.
			return "the default route", true
		}
		models = append(models, c.DefaultModel)
	}
	for _, model := range models {
		if c.isCallerModel(model) {
			return model, true
		}
	}
	return "", false
}

func (c *RouterConfig) isCallerModel(model string) bool {
	_, mode, ok := c.CredentialBackendOf(model)
	return ok && mode == CredentialCaller
}

// firstCallerBackend returns the path of the first caller backend, if any.
func (c *RouterConfig) firstCallerBackend() (string, bool) {
	for _, model := range sortedModelKeys(c.ModelConfig) {
		if path, mode, ok := c.CredentialBackendOf(model); ok && mode == CredentialCaller {
			return path, true
		}
	}
	return "", false
}

// CredentialBackendOf returns the path and credential mode of the first
// backend of a provider model that takes a credential the request brings
// (delegated or caller), and whether it has one.
func (c *RouterConfig) CredentialBackendOf(model string) (string, string, bool) {
	params, ok := c.ModelConfig[model]
	if !ok {
		return "", "", false
	}
	for index, endpoint := range params.PreferredEndpoints {
		profile, err := c.GetProviderProfileForEndpoint(endpoint)
		if err == nil && (profile.Delegated() || profile.Caller()) {
			return fmt.Sprintf("providers.models[%s].backend_refs[%d]", model, index), profile.AuthCredential, true
		}
	}
	return "", "", false
}

func isLoopbackAddress(address string) bool {
	host := strings.Trim(strings.TrimSpace(address), "[]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// HasCallerBackends reports whether any provider backend takes the caller's
// login, which is when the Router keeps it from a client request.
func (c *RouterConfig) HasCallerBackends() bool {
	if c == nil {
		return false
	}
	for _, profile := range c.ProviderProfiles {
		if profile.Caller() {
			return true
		}
	}
	return false
}
