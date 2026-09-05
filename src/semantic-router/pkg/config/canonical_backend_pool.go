package config

import (
	"fmt"
	"maps"
	"net"
	"net/url"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// effectiveBackendSemantics contains every request or transport property that
// the current one-route/one-cluster Envoy projection cannot vary after endpoint
// load balancing has selected an upstream.
type effectiveBackendSemantics struct {
	providerID         string
	protocol           string
	modelID            string
	transport          string
	discovery          string
	basePath           string
	tlsServerName      string
	credentials        effectiveCredentialIdentity
	authHeader         string
	authPrefix         string
	headers            map[string]string
	apiVersion         string
	chatPath           string
	createPath         string
	reasoningTransport modelcatalog.ReasoningTransport
}

type effectiveCredentialIdentity struct {
	source string
	value  string
}

type effectiveBackendCandidate struct {
	bindingIndex int
	semantics    effectiveBackendSemantics
}

func validateEffectiveBackendPools(models []modelcatalog.EffectiveModel) error {
	for _, model := range models {
		if err := validateEffectiveBackendPool(model); err != nil {
			return err
		}
	}
	return nil
}

func validateEffectiveBackendPool(model modelcatalog.EffectiveModel) error {
	candidates, err := effectiveBackendCandidates(model)
	if err != nil {
		return err
	}
	if len(candidates) < 2 {
		return nil
	}
	baseline := candidates[0]
	for _, current := range candidates[1:] {
		differences := effectiveBackendDifferences(baseline.semantics, current.semantics)
		if len(differences) == 0 {
			continue
		}
		return fmt.Errorf(
			"providers.models[%s].backend_refs[%d] cannot share one Envoy cluster with backend_refs[%d]: %s differ; split heterogeneous backends into separate model aliases so provider metadata follows the selected upstream",
			model.Alias,
			current.bindingIndex,
			baseline.bindingIndex,
			strings.Join(differences, ", "),
		)
	}
	return nil
}

func effectiveBackendCandidates(model modelcatalog.EffectiveModel) ([]effectiveBackendCandidate, error) {
	var result []effectiveBackendCandidate
	for bindingIndex, provider := range model.Providers {
		for _, endpoint := range effectiveProviderURLs(provider.Provider) {
			semantics, err := newEffectiveBackendSemantics(provider, endpoint.url)
			if err != nil {
				return nil, fmt.Errorf(
					"providers.models[%s].backend_refs[%d]: %w",
					model.Alias,
					bindingIndex,
					err,
				)
			}
			result = append(result, effectiveBackendCandidate{
				bindingIndex: bindingIndex,
				semantics:    semantics,
			})
		}
	}
	return result, nil
}

func newEffectiveBackendSemantics(
	effective modelcatalog.EffectiveModelProvider,
	endpointURL string,
) (effectiveBackendSemantics, error) {
	host, _, scheme, err := endpointAddress(endpointURL)
	if err != nil {
		return effectiveBackendSemantics{}, err
	}
	parsed, err := url.Parse(endpointURLForParsing(endpointURL))
	if err != nil {
		return effectiveBackendSemantics{}, fmt.Errorf("invalid provider endpoint URL %q", endpointURL)
	}
	// Template-bearing maintained assets are validated before deployment
	// substitution. The host is irrelevant to path/auth comparison, so reuse
	// the same placeholder-safe URL form as endpoint parsing.
	profile := materializedProviderProfile(effective, endpointURLForParsing(endpointURL))
	authHeader, authPrefix, err := profile.ResolveAuthHeader()
	if err != nil {
		return effectiveBackendSemantics{}, err
	}
	reasoningTransport, err := profile.ResolveReasoningTransport()
	if err != nil {
		return effectiveBackendSemantics{}, err
	}
	createPath, err := profile.ResolveCreatePath(effective.Binding.Protocol)
	if err != nil {
		return effectiveBackendSemantics{}, err
	}
	discovery := "dns"
	if net.ParseIP(host) != nil {
		discovery = "ip"
	}
	tlsServerName := ""
	if scheme == "https" {
		tlsServerName = host
	}
	return effectiveBackendSemantics{
		providerID:         effective.Provider.Definition.ID,
		protocol:           effective.Binding.Protocol,
		modelID:            effective.Binding.ModelID,
		transport:          scheme,
		discovery:          discovery,
		basePath:           strings.TrimSuffix(parsed.Path, "/"),
		tlsServerName:      tlsServerName,
		credentials:        effectiveBackendCredential(effective.Provider.Instance.Credentials),
		authHeader:         authHeader,
		authPrefix:         authPrefix,
		headers:            profile.ExtraHeaders,
		apiVersion:         effective.Provider.Instance.APIVersion,
		chatPath:           effective.Provider.Instance.ChatPath,
		createPath:         createPath,
		reasoningTransport: reasoningTransport,
	}, nil
}

func effectiveBackendCredential(credentials modelcatalog.CredentialsRef) effectiveCredentialIdentity {
	if credentials.APIKey != "" {
		return effectiveCredentialIdentity{source: "inline", value: credentials.APIKey}
	}
	if credentials.APIKeyEnv != "" {
		return effectiveCredentialIdentity{source: "environment", value: credentials.APIKeyEnv}
	}
	return effectiveCredentialIdentity{}
}

func effectiveBackendDifferences(left, right effectiveBackendSemantics) []string {
	var result []string
	result = appendDifference(result, left.providerID != right.providerID, "provider")
	result = appendDifference(result, left.protocol != right.protocol, "wire protocol")
	result = appendDifference(result, left.modelID != right.modelID, "native model ID")
	result = appendDifference(result, left.transport != right.transport, "URL scheme")
	result = appendDifference(result, left.discovery != right.discovery, "DNS/IP discovery")
	result = appendDifference(result, left.basePath != right.basePath, "base path")
	result = appendDifference(result, left.tlsServerName != right.tlsServerName, "TLS server name")
	result = appendDifference(result, left.credentials != right.credentials, "credential source")
	result = appendDifference(result, left.authHeader != right.authHeader, "auth header")
	result = appendDifference(result, left.authPrefix != right.authPrefix, "auth prefix")
	result = appendDifference(result, !maps.Equal(left.headers, right.headers), "extra headers")
	result = appendDifference(result, left.apiVersion != right.apiVersion, "API version")
	result = appendDifference(result, left.chatPath != right.chatPath, "chat path")
	result = appendDifference(result, left.createPath != right.createPath, "request path")
	result = appendDifference(
		result,
		left.reasoningTransport != right.reasoningTransport,
		"reasoning transport",
	)
	return result
}

func appendDifference(result []string, differs bool, label string) []string {
	if differs {
		return append(result, label)
	}
	return result
}
