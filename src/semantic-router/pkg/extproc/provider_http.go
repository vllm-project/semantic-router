package extproc

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
)

// splitProviderEndpoint parses a resolved provider endpoint into the absolute
// path and the query the connector sends separately. A provider profile may
// append a query, as Azure OpenAI does with api-version, and an operator's
// chat_path override is free text, so the value is validated here rather than
// trusted: it must be a path-only reference with no scheme, host, or fragment,
// and any query must parse.
func splitProviderEndpoint(endpoint string) (string, string, error) {
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return "", "", fmt.Errorf("parse endpoint path %q: %w", endpoint, err)
	}
	if parsed.Scheme != "" || parsed.Host != "" || parsed.User != nil || parsed.Opaque != "" {
		return "", "", fmt.Errorf("endpoint path %q must not name a scheme or host", endpoint)
	}
	if parsed.Fragment != "" || parsed.RawFragment != "" || strings.Contains(endpoint, "#") {
		return "", "", fmt.Errorf("endpoint path %q must not carry a fragment", endpoint)
	}
	if !strings.HasPrefix(parsed.Path, "/") {
		return "", "", fmt.Errorf("endpoint path %q must be absolute", endpoint)
	}
	if _, err := url.ParseQuery(parsed.RawQuery); err != nil {
		return "", "", fmt.Errorf("endpoint path %q has an invalid query: %w", endpoint, err)
	}
	return parsed.Path, parsed.RawQuery, nil
}

func providerEndpointScheme(cfg *config.RouterConfig, backendName string, profile *config.ProviderProfile) string {
	if profile != nil && profile.BaseURL != "" {
		if parsed, err := url.Parse(profile.BaseURL); err == nil && parsed.Scheme != "" {
			return parsed.Scheme
		}
		return "http"
	}
	if endpoint, ok := cfg.GetEndpointByName(backendName); ok && endpoint != nil &&
		strings.EqualFold(strings.TrimSpace(endpoint.Protocol), "https") {
		return "https"
	}
	return "http"
}

// providerTLSConfig is the TLS client configuration of a Router-initiated
// call to a backend: its tls block, so the Router trusts the CA, server name
// and client certificate the backend's Envoy cluster uses. nil keeps the
// system roots and the endpoint's host.
func providerTLSConfig(profile *config.ProviderProfile) (*tls.Config, error) {
	if profile == nil {
		return nil, nil
	}
	return profile.TLS.ClientConfig()
}

// providerEndpointPath mirrors primary dispatch for router-initiated provider calls.
func providerEndpointPath(profile *config.ProviderProfile, format llmprotocol.WireFormat) string {
	path := requestWirePath(format)
	if profile == nil {
		return path
	}
	if configured, err := profile.ResolveCreatePath(requestWireProtocol(format)); err == nil && configured != "" {
		return configured
	}
	return path
}

// configuredProviderAuthorizer uses only configured backend credentials.
// Router-own preprocessing, probes and shadow calls never forward client
// keys, and a delegated backend refuses them: they have no user's credential.
// Like request routing, they send no key to a provider whose auth is none.
func configuredProviderAuthorizer(
	cfg *config.RouterConfig,
	profile *config.ProviderProfile,
	model string,
	call string,
) (func(context.Context, *http.Request) error, error) {
	provider, providerAuth, err := resolveProviderAuth(profile)
	if err != nil {
		return nil, err
	}
	if profile.Delegated() || profile.Caller() {
		logging.ComponentWarnEvent("extproc", "delegated_router_call_refused", map[string]interface{}{
			"call": call, "model": model, "credential": profile.AuthCredential,
		})
		return nil, principal.ErrRouterCall
	}
	if cfg == nil || providerAuth.Strategy == "none" {
		return nil, nil
	}
	return func(_ context.Context, request *http.Request) error {
		accessKey := authz.NewStaticConfigProvider(cfg).GetKey(provider, model, nil)
		if accessKey == "" {
			return nil
		}
		if providerAuth.Prefix != "" {
			accessKey = providerAuth.Prefix + " " + accessKey
		}
		request.Header.Set(providerAuth.Header, accessKey)
		return nil
	}, nil
}
