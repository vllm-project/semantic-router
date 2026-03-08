package authoring

import (
	"fmt"
	"net"
	"net/url"
	"strconv"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func compileModel(model Model) (routerconfig.ModelParams, []routerconfig.VLLMEndpoint, error) {
	params := routerconfig.ModelParams{
		PreferredEndpoints: make([]string, 0, len(model.Endpoints)),
		ReasoningFamily:    model.ReasoningFamily,
		AccessKey:          model.AccessKey,
		ParamSize:          model.ParamSize,
		APIFormat:          model.APIFormat,
		Description:        model.Description,
		Capabilities:       append([]string(nil), model.Capabilities...),
		QualityScore:       model.QualityScore,
	}
	if model.Pricing != nil {
		params.Pricing = routerconfig.ModelPricing{
			Currency:        model.Pricing.Currency,
			PromptPer1M:     model.Pricing.PromptPer1M,
			CompletionPer1M: model.Pricing.CompletionPer1M,
		}
	}

	if len(model.Endpoints) == 0 {
		return params, nil, nil
	}

	endpoints := make([]routerconfig.VLLMEndpoint, 0, len(model.Endpoints))
	for _, endpoint := range model.Endpoints {
		if strings.TrimSpace(endpoint.Name) == "" {
			return routerconfig.ModelParams{}, nil, fmt.Errorf("providers.models[%q].endpoints[].name is required", model.Name)
		}
		address, port, protocol, err := parseEndpoint(endpoint.Endpoint, endpoint.Protocol)
		if err != nil {
			return routerconfig.ModelParams{}, nil, fmt.Errorf("providers.models[%q].endpoints[%q]: %w", model.Name, endpoint.Name, err)
		}

		runtimeName := compiledEndpointName(model.Name, endpoint.Name)
		params.PreferredEndpoints = append(params.PreferredEndpoints, runtimeName)
		endpoints = append(endpoints, routerconfig.VLLMEndpoint{
			Name:     runtimeName,
			Address:  address,
			Port:     port,
			Weight:   endpoint.Weight,
			Model:    model.Name,
			Protocol: protocol,
		})
	}

	return params, endpoints, nil
}

func compileReasoningFamilies(families map[string]ReasoningFamily) map[string]routerconfig.ReasoningFamilyConfig {
	if len(families) == 0 {
		return nil
	}

	compiled := make(map[string]routerconfig.ReasoningFamilyConfig, len(families))
	for name, family := range families {
		compiled[name] = routerconfig.ReasoningFamilyConfig{
			Type:      family.Type,
			Parameter: family.Parameter,
		}
	}
	return compiled
}

func compiledEndpointName(modelName, endpointName string) string {
	return modelName + "_" + endpointName
}

func parseEndpoint(rawEndpoint, protocolHint string) (string, int, string, error) {
	endpoint := strings.TrimSpace(rawEndpoint)
	if endpoint == "" {
		return "", 0, "", fmt.Errorf("endpoint is required")
	}

	if strings.Contains(endpoint, "://") {
		return parseURLEndpoint(endpoint)
	}

	protocol := normalizeProtocol(protocolHint)
	hostPort := stripEndpointPath(endpoint)
	if hostPort == "" {
		return "", 0, "", fmt.Errorf("endpoint host is required")
	}
	if strings.HasPrefix(hostPort, "[") && strings.Contains(hostPort, "]:") {
		return parseBracketedHostPort(rawEndpoint, hostPort, protocol)
	}
	if strings.Count(hostPort, ":") == 1 {
		return parseNamedHostPort(rawEndpoint, hostPort, protocol)
	}
	return hostPort, defaultPort(protocol), protocol, nil
}

func parseURLEndpoint(endpoint string) (string, int, string, error) {
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return "", 0, "", fmt.Errorf("parse endpoint URL: %w", err)
	}
	if parsed.Hostname() == "" {
		return "", 0, "", fmt.Errorf("endpoint host is required")
	}

	protocol := normalizeProtocol(parsed.Scheme)
	port := parsed.Port()
	if port == "" {
		return parsed.Hostname(), defaultPort(protocol), protocol, nil
	}

	parsedPort, err := strconv.Atoi(port)
	if err != nil {
		return "", 0, "", fmt.Errorf("invalid endpoint port %q", port)
	}
	return parsed.Hostname(), parsedPort, protocol, nil
}

func stripEndpointPath(endpoint string) string {
	if slash := strings.Index(endpoint, "/"); slash >= 0 {
		return endpoint[:slash]
	}
	return endpoint
}

func parseBracketedHostPort(rawEndpoint, hostPort, protocol string) (string, int, string, error) {
	host, port, err := net.SplitHostPort(hostPort)
	if err != nil {
		return "", 0, "", fmt.Errorf("invalid endpoint address %q: %w", rawEndpoint, err)
	}
	parsedPort, err := strconv.Atoi(port)
	if err != nil {
		return "", 0, "", fmt.Errorf("invalid endpoint port %q", port)
	}
	return host, parsedPort, protocol, nil
}

func parseNamedHostPort(rawEndpoint, hostPort, protocol string) (string, int, string, error) {
	host, port, found := strings.Cut(hostPort, ":")
	if !found || strings.TrimSpace(host) == "" || strings.TrimSpace(port) == "" {
		return "", 0, "", fmt.Errorf("invalid endpoint address %q", rawEndpoint)
	}
	parsedPort, err := strconv.Atoi(port)
	if err != nil {
		return "", 0, "", fmt.Errorf("invalid endpoint port %q", port)
	}
	return host, parsedPort, protocol, nil
}

func normalizeProtocol(protocol string) string {
	switch strings.ToLower(strings.TrimSpace(protocol)) {
	case "", "http":
		return "http"
	case "https":
		return "https"
	default:
		return strings.ToLower(strings.TrimSpace(protocol))
	}
}

func defaultPort(protocol string) int {
	if protocol == "https" {
		return 443
	}
	return 80
}
