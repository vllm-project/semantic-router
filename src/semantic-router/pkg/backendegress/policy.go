// Package backendegress enforces the immutable operator egress boundary shared
// by Model validation, discovery, probes, and inference dispatch.
package backendegress

import (
	"bytes"
	"fmt"
	"net"
	"net/netip"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

const (
	policyVersion      = "v1"
	maximumPolicyBytes = 1 << 20
	maximumResolvedIPs = 64
	minimumAllowedPort = 1
	maximumAllowedPort = 65535
)

type Config struct {
	Version string       `yaml:"version"`
	Schemes []string     `yaml:"schemes"`
	Hosts   []HostConfig `yaml:"hosts"`
}

type HostConfig struct {
	Host       string   `yaml:"host"`
	Ports      []uint16 `yaml:"ports"`
	AllowCIDRs []string `yaml:"allow_cidrs,omitempty"`
}

type Policy struct {
	schemes map[string]struct{}
	rules   []hostRule
}

type hostRule struct {
	host         string
	wildcard     bool
	ports        map[uint16]struct{}
	privateCIDRs []netip.Prefix
}

type Target struct {
	Origin     string
	Scheme     string
	Host       string
	Port       uint16
	ServerName string
	rule       hostRule
}

func LoadFile(path string) (Policy, error) {
	info, err := os.Stat(path)
	if err != nil {
		return Policy{}, fmt.Errorf("stat backend egress policy: %w", err)
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > maximumPolicyBytes {
		return Policy{}, fmt.Errorf("backend egress policy must be a non-empty regular file at most %d bytes", maximumPolicyBytes)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Policy{}, fmt.Errorf("read backend egress policy: %w", err)
	}
	return Parse(data)
}

func Parse(data []byte) (Policy, error) {
	if len(data) == 0 || len(data) > maximumPolicyBytes {
		return Policy{}, fmt.Errorf("backend egress policy size is invalid")
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	var config Config
	if err := decoder.Decode(&config); err != nil {
		return Policy{}, fmt.Errorf("decode backend egress policy: %w", err)
	}
	return Compile(config)
}

func Compile(config Config) (Policy, error) {
	if config.Version != policyVersion {
		return Policy{}, fmt.Errorf("backend egress policy version must be %q", policyVersion)
	}
	policy := Policy{schemes: make(map[string]struct{}, len(config.Schemes)), rules: make([]hostRule, 0, len(config.Hosts))}
	for _, scheme := range config.Schemes {
		if scheme != "http" && scheme != "https" {
			return Policy{}, fmt.Errorf("backend egress scheme %q is unsupported", scheme)
		}
		if _, duplicate := policy.schemes[scheme]; duplicate {
			return Policy{}, fmt.Errorf("backend egress scheme %q is duplicated", scheme)
		}
		policy.schemes[scheme] = struct{}{}
	}
	if len(policy.schemes) == 0 || len(config.Hosts) == 0 {
		return Policy{}, fmt.Errorf("backend egress policy requires schemes and hosts")
	}
	identities := make(map[string]struct{}, len(config.Hosts))
	for index, raw := range config.Hosts {
		rule, err := compileHostRule(raw)
		if err != nil {
			return Policy{}, fmt.Errorf("backend egress host %d: %w", index, err)
		}
		identity := rule.host
		if rule.wildcard {
			identity = "*." + identity
		}
		if _, duplicate := identities[identity]; duplicate {
			return Policy{}, fmt.Errorf("backend egress host %q is duplicated", identity)
		}
		identities[identity] = struct{}{}
		policy.rules = append(policy.rules, rule)
	}
	sortHostRules(policy.rules)
	return policy, nil
}

// Overlay returns a new policy where exact host identities from overrides
// replace the corresponding base rules. Override schemes must already be
// allowed by the base policy, preventing a narrow system exception from
// widening the transport contract for unrelated hosts.
func Overlay(base, overrides Policy) (Policy, error) {
	result := Policy{
		schemes: make(map[string]struct{}, len(base.schemes)),
		rules:   make([]hostRule, 0, len(base.rules)+len(overrides.rules)),
	}
	for scheme := range base.schemes {
		result.schemes[scheme] = struct{}{}
	}
	for scheme := range overrides.schemes {
		if _, allowed := base.schemes[scheme]; !allowed {
			return Policy{}, fmt.Errorf("backend egress override scheme %q is not allowed by the base policy", scheme)
		}
	}
	overridden := make(map[string]struct{}, len(overrides.rules))
	for _, rule := range overrides.rules {
		if rule.wildcard {
			return Policy{}, fmt.Errorf("backend egress overrides require exact host identities")
		}
		overridden[hostRuleIdentity(rule)] = struct{}{}
	}
	for _, rule := range base.rules {
		if _, replace := overridden[hostRuleIdentity(rule)]; replace {
			continue
		}
		result.rules = append(result.rules, cloneHostRule(rule))
	}
	for _, rule := range overrides.rules {
		result.rules = append(result.rules, cloneHostRule(rule))
	}
	sortHostRules(result.rules)
	return result, nil
}

func cloneHostRule(rule hostRule) hostRule {
	ports := make(map[uint16]struct{}, len(rule.ports))
	for port := range rule.ports {
		ports[port] = struct{}{}
	}
	return hostRule{
		host:         rule.host,
		wildcard:     rule.wildcard,
		ports:        ports,
		privateCIDRs: append([]netip.Prefix(nil), rule.privateCIDRs...),
	}
}

func hostRuleIdentity(rule hostRule) string {
	if rule.wildcard {
		return "*." + rule.host
	}
	return rule.host
}

func sortHostRules(rules []hostRule) {
	sort.Slice(rules, func(left, right int) bool {
		if rules[left].wildcard != rules[right].wildcard {
			return !rules[left].wildcard
		}
		return len(rules[left].host) > len(rules[right].host)
	})
}

func compileHostRule(config HostConfig) (hostRule, error) {
	host := strings.ToLower(strings.TrimSpace(config.Host))
	wildcard := strings.HasPrefix(host, "*.")
	if wildcard {
		host = strings.TrimPrefix(host, "*.")
	}
	if host == "" || strings.Contains(host, "*") || strings.TrimSpace(config.Host) != config.Host {
		return hostRule{}, fmt.Errorf("host pattern is invalid")
	}
	if parsed := net.ParseIP(host); parsed == nil && !validDNSName(host, !wildcard) {
		return hostRule{}, fmt.Errorf("host name is invalid")
	} else if parsed != nil && wildcard {
		return hostRule{}, fmt.Errorf("IP literals cannot use wildcards")
	}
	ports := make(map[uint16]struct{}, len(config.Ports))
	for _, port := range config.Ports {
		if port < minimumAllowedPort || port > maximumAllowedPort {
			return hostRule{}, fmt.Errorf("port is invalid")
		}
		if _, duplicate := ports[port]; duplicate {
			return hostRule{}, fmt.Errorf("port %d is duplicated", port)
		}
		ports[port] = struct{}{}
	}
	if len(ports) == 0 {
		return hostRule{}, fmt.Errorf("at least one port is required")
	}
	privateCIDRs := make([]netip.Prefix, 0, len(config.AllowCIDRs))
	for _, raw := range config.AllowCIDRs {
		prefix, err := netip.ParsePrefix(raw)
		if err != nil || prefix.String() != raw {
			return hostRule{}, fmt.Errorf("private-network exception %q is not canonical", raw)
		}
		prefix = prefix.Masked()
		if !safePrivateException(prefix) {
			return hostRule{}, fmt.Errorf("private-network exception %q is unsafe or not private", raw)
		}
		privateCIDRs = append(privateCIDRs, prefix)
	}
	return hostRule{host: host, wildcard: wildcard, ports: ports, privateCIDRs: privateCIDRs}, nil
}

func safePrivateException(candidate netip.Prefix) bool {
	contained := false
	for _, private := range privatePrefixes {
		if candidate.Addr().BitLen() == private.Addr().BitLen() &&
			private.Contains(candidate.Addr()) && candidate.Bits() >= private.Bits() {
			contained = true
			break
		}
	}
	if !contained {
		return false
	}
	for _, blocked := range criticalPrefixes {
		if candidate.Addr().BitLen() != blocked.Addr().BitLen() {
			continue
		}
		if candidate.Contains(blocked.Addr()) || blocked.Contains(candidate.Addr()) {
			return false
		}
	}
	return true
}

var privatePrefixes = []netip.Prefix{
	netip.MustParsePrefix("10.0.0.0/8"),
	netip.MustParsePrefix("172.16.0.0/12"),
	netip.MustParsePrefix("192.168.0.0/16"),
	netip.MustParsePrefix("fc00::/7"),
}

func (p Policy) AuthorizeOrigin(origin string) (Target, error) {
	normalized, err := providercredential.NormalizeOrigin(origin)
	if err != nil || normalized != origin {
		return Target{}, fmt.Errorf("backend origin is not canonical")
	}
	parsed, _ := url.Parse(origin)
	if _, allowed := p.schemes[parsed.Scheme]; !allowed {
		return Target{}, fmt.Errorf("backend origin scheme is denied")
	}
	host := strings.ToLower(parsed.Hostname())
	rule, found := p.ruleForHost(host)
	if !found {
		return Target{}, fmt.Errorf("backend origin host is denied")
	}
	port := uint16(443)
	if parsed.Scheme == "http" {
		port = 80
	}
	if parsed.Port() != "" {
		value, parseErr := strconv.ParseUint(parsed.Port(), 10, 16)
		if parseErr != nil || value == 0 {
			return Target{}, fmt.Errorf("backend origin port is invalid")
		}
		port = uint16(value)
	}
	if _, allowed := rule.ports[port]; !allowed {
		return Target{}, fmt.Errorf("backend origin port is denied")
	}
	serverName := host
	if net.ParseIP(host) != nil {
		serverName = ""
	}
	return Target{Origin: origin, Scheme: parsed.Scheme, Host: host, Port: port, ServerName: serverName, rule: rule}, nil
}

func (p Policy) ruleForHost(host string) (hostRule, bool) {
	for _, rule := range p.rules {
		if !rule.wildcard && host == rule.host {
			return rule, true
		}
		if rule.wildcard && strings.HasSuffix(host, "."+rule.host) && host != rule.host {
			return rule, true
		}
	}
	return hostRule{}, false
}

func validDNSName(host string, allowServiceLabels bool) bool {
	if len(host) > 253 || strings.HasPrefix(host, ".") || strings.HasSuffix(host, ".") || strings.Contains(host, "..") {
		return false
	}
	for _, label := range strings.Split(host, ".") {
		if len(label) == 0 || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
			return false
		}
		for _, char := range label {
			if char != '-' && (!allowServiceLabels || char != '_') &&
				(char < 'a' || char > 'z') && (char < '0' || char > '9') {
				return false
			}
		}
	}
	return true
}
