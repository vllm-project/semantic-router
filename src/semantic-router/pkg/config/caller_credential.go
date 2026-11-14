package config

import (
	"fmt"
	"net"
	"slices"
	"strings"
)

// A provider backend whose auth is credential: caller takes the provider
// login the client itself sent the Router, as Claude Code sends its Claude
// login, so the gateway routes a person's own subscription without holding
// a key. Only the client's own request presents it: a call the Router makes
// on its own behalf, an in-process hop and a response cache have no login of
// that person's, so each fails to load. The first version keeps the payer
// and the provider single: once one backend takes the caller's login, every
// backend does, and each is Anthropic's.

// CallerCredentialProviders are the providers a caller backend may name: the
// provider whose login the client sends.
var CallerCredentialProviders = []string{"anthropic"}

// CallerCredentialHost is the only remote host a caller backend may name: the
// API that issued the login.
const CallerCredentialHost = "api.anthropic.com" // #nosec G101 -- a host name, not a credential

// validateCallerUpstreams checks every caller backend against what the
// rest of the configuration would do with a person's login.
func validateCallerUpstreams(cfg *RouterConfig) error {
	first, ok := cfg.firstCallerBackend()
	if !ok {
		return nil
	}
	for _, model := range sortedModelKeys(cfg.ModelConfig) {
		for index, endpoint := range cfg.ModelConfig[model].PreferredEndpoints {
			path := fmt.Sprintf("providers.models[%s].backend_refs[%d]", model, index)
			profile, err := cfg.GetProviderProfileForEndpoint(endpoint)
			if err != nil || !profile.Caller() {
				return fmt.Errorf("%s: takes a key while %s takes the caller's login; a request would change who pays when it is routed or falls back, so every backend takes auth.credential: caller", path, first)
			}
			if err := validateCallerBackend(cfg, path, endpoint, profile); err != nil {
				return err
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
	provider := strings.ToLower(strings.TrimSpace(profile.Type))
	if !slices.Contains(CallerCredentialProviders, provider) {
		return fmt.Errorf("%s: auth.credential: caller sends the client's login, which only %s issues; provider %q would receive a login it did not issue", path, strings.Join(CallerCredentialProviders, ", "), profile.Type)
	}
	for _, endpoint := range cfg.VLLMEndpoints {
		if endpoint.Name != endpointName {
			continue
		}
		if isLoopbackAddress(endpoint.Address) {
			continue
		}
		if !strings.EqualFold(endpoint.Protocol, "https") || !strings.EqualFold(strings.TrimSuffix(endpoint.Address, "."), CallerCredentialHost) {
			return fmt.Errorf("%s: auth.credential: caller sends the client's login, so the backend is https://%s, or an endpoint on this machine", path, CallerCredentialHost)
		}
	}
	return nil
}

// validateCallerRequestOnly refuses whatever would call a caller backend
// without the client's own request, an in-process hop and an answer kept in
// a response cache for other requests, and a system prompt the Router writes
// over the client's, which Anthropic refuses on a person's login.
func (c *RouterConfig) validateCallerRequestOnly(first string) error {
	recipes := c.Recipes
	if len(recipes) == 0 {
		recipes = []RoutingRecipe{*c.DefaultRecipe()}
	}
	if ResponseCachePluginOf(c.GlobalPlugins) != nil {
		return fmt.Errorf("global.plugins: response_cache keeps answers for other requests, and %s takes the caller's login, which names no user to keep them for; remove response_cache", first)
	}
	for _, recipe := range recipes {
		if ResponseCachePluginOf(recipe.Profile.DefaultPlugins) != nil {
			return fmt.Errorf("recipes[%s].plugins: response_cache keeps answers for other requests, and %s takes the caller's login; remove response_cache", recipe.Name, first)
		}
		for _, decision := range recipe.Profile.Decisions {
			path := fmt.Sprintf("recipes[%s].decisions[%s]", recipe.Name, decision.Name)
			if ResponseCachePluginOf(decision.DeclaredPlugins()) != nil {
				return fmt.Errorf("%s.plugins: response_cache keeps answers for other requests, and %s takes the caller's login; remove response_cache", path, first)
			}
			if prompt := decision.GetSystemPromptConfig(); prompt.IsEnabled() {
				return fmt.Errorf("%s.plugins: system_prompt (mode %s) changes how the client's system prompt starts, and Anthropic refuses a request on a person's login whose system prompt does not start with the client's own; %s takes the caller's login, remove system_prompt", path, prompt.EffectiveMode(), first)
			}
			if called := decisionCalledModels(decision); len(called) > 0 {
				return fmt.Errorf("%s: the Router calls %q in process, and a hop never carries the caller's login that %s takes; route this decision to one model", path, called[0], first)
			}
		}
	}
	return nil
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
