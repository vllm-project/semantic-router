// Package agenttoolsource owns the untrusted remote Tool Source boundary.
// It composes source-scoped policy with the Router's immutable operator egress
// policy and never exposes credentials to the Agent domain.
package agenttoolsource

import (
	"fmt"
	"net"
	"net/url"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
)

// PolicyCompiler is both the Management mutation validator and the execution
// policy factory. Both paths use backendegress.Compile, so a stored Tool Source
// can never rely on a weaker, parallel SSRF allow-list implementation.
type PolicyCompiler struct{}

func (PolicyCompiler) Normalize(input agentmanagement.ToolSourceInput) (agentmanagement.ToolSourceInput, error) {
	input, err := agentmanagement.NormalizeToolSourceInput(input)
	if err != nil {
		return agentmanagement.ToolSourceInput{}, err
	}
	parsed, _ := url.Parse(input.Endpoint)
	policy, normalized, err := compilePolicy(parsed, input.EgressPolicy)
	if err != nil {
		return agentmanagement.ToolSourceInput{}, err
	}
	origin := parsed.Scheme + "://" + parsed.Host
	if _, err := policy.AuthorizeOrigin(origin); err != nil {
		return agentmanagement.ToolSourceInput{}, fmt.Errorf("%w: Tool Source endpoint is denied by its egress policy", agentmanagement.ErrDenied)
	}
	input.EgressPolicy = normalized
	return input, nil
}

func (PolicyCompiler) Compile(policy agentmanagement.EgressPolicy) (backendegress.Policy, error) {
	compiled, _, err := compilePolicy(nil, policy)
	return compiled, err
}

func compilePolicy(
	endpoint *url.URL,
	input agentmanagement.EgressPolicy,
) (backendegress.Policy, agentmanagement.EgressPolicy, error) {
	hosts, err := canonicalHosts(input.AllowedHosts)
	if err != nil {
		return backendegress.Policy{}, agentmanagement.EgressPolicy{}, err
	}
	ports, err := canonicalPorts(input.AllowedPorts)
	if err != nil {
		return backendegress.Policy{}, agentmanagement.EgressPolicy{}, err
	}
	cidrs := canonicalStrings(input.AllowedPrivateCIDRs)
	rules := make([]backendegress.HostConfig, 0, len(hosts))
	for _, host := range hosts {
		rulePorts := make([]uint16, len(ports))
		for index, port := range ports {
			rulePorts[index] = uint16(port)
		}
		rules = append(rules, backendegress.HostConfig{
			Host: host, Ports: rulePorts, AllowCIDRs: append([]string(nil), cidrs...),
		})
	}
	compiled, err := backendegress.Compile(backendegress.Config{
		Version: "v1", Schemes: []string{"https"}, Hosts: rules,
	})
	if err != nil {
		return backendegress.Policy{}, agentmanagement.EgressPolicy{},
			fmt.Errorf("%w: Tool Source egress policy: %w", agentmanagement.ErrDenied, err)
	}
	if endpoint != nil {
		host := strings.ToLower(endpoint.Hostname())
		port := 443
		if endpoint.Port() != "" {
			parsedPort, parseErr := strconv.Atoi(endpoint.Port())
			if parseErr != nil {
				return backendegress.Policy{}, agentmanagement.EgressPolicy{}, agentmanagement.ErrInvalid
			}
			port = parsedPort
		}
		if !contains(hosts, host) || !containsPort(ports, port) {
			return backendegress.Policy{}, agentmanagement.EgressPolicy{},
				fmt.Errorf("%w: Tool Source endpoint is outside its egress policy", agentmanagement.ErrDenied)
		}
	}
	return compiled, agentmanagement.EgressPolicy{
		AllowedHosts: hosts, AllowedPorts: ports, AllowedPrivateCIDRs: cidrs,
	}, nil
}

func canonicalHosts(values []string) ([]string, error) {
	if len(values) == 0 || len(values) > 64 {
		return nil, agentmanagement.ErrInvalid
	}
	result := canonicalStrings(values)
	if len(result) == 0 {
		return nil, agentmanagement.ErrInvalid
	}
	for _, host := range result {
		if host != strings.ToLower(host) || strings.HasSuffix(host, ".") ||
			(strings.HasPrefix(host, "*.") && net.ParseIP(strings.TrimPrefix(host, "*.")) != nil) {
			return nil, fmt.Errorf("%w: Tool Source allowed host is not canonical", agentmanagement.ErrInvalid)
		}
	}
	return result, nil
}

func canonicalPorts(values []int) ([]int, error) {
	if len(values) == 0 {
		return []int{443}, nil
	}
	if len(values) > 32 {
		return nil, agentmanagement.ErrInvalid
	}
	result := append([]int(nil), values...)
	sort.Ints(result)
	write := 0
	for _, port := range result {
		if port < 1 || port > 65535 {
			return nil, fmt.Errorf("%w: Tool Source port is invalid", agentmanagement.ErrInvalid)
		}
		if write > 0 && result[write-1] == port {
			continue
		}
		result[write] = port
		write++
	}
	return result[:write], nil
}

func canonicalStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, duplicate := seen[value]; duplicate {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func contains(values []string, candidate string) bool {
	index := sort.SearchStrings(values, candidate)
	return index < len(values) && values[index] == candidate
}

func containsPort(values []int, candidate int) bool {
	index := sort.SearchInts(values, candidate)
	return index < len(values) && values[index] == candidate
}

var _ agentmanagement.ToolSourcePolicyValidator = PolicyCompiler{}
