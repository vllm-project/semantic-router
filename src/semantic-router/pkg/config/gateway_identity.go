package config

import (
	"fmt"
	"strconv"
	"strings"
)

// CapabilityGatewayIdentity is policy that enforces access by a client
// identity asserted in request headers. A standalone Router drops client-sent
// identity headers unless a listener trusts them, so without one that policy
// could never match.
const CapabilityGatewayIdentity = "gateway_asserted_identity"

func init() {
	GatewayCapabilities.MustRegister(CapabilityGatewayIdentity, GatewayCapability{
		Modes: []GatewayMode{GatewayExtProc},
		Uses:  gatewayIdentityUses,
		Unserved: "needs an identity source in standalone mode; behind an authenticating proxy, " +
			"set listeners[].identity.trust_headers: true, or serve with --gateway extproc",
	})
}

func gatewayIdentityUses(cfg *RouterConfig) []CapabilityUse {
	if cfg.TrustsIdentityHeaders() {
		return nil
	}
	reads := fmt.Sprintf("reads the client identity an authenticating gateway sets in %s and %s",
		cfg.Authz.Identity.GetUserIDHeader(), cfg.Authz.Identity.GetUserGroupsHeader())
	var uses []CapabilityUse
	for _, at := range cfg.RoutingDecisionsAt() {
		if name, ok := firstSignalOfType(&at.Decision.Rules, SignalTypeAuthz); ok {
			uses = append(uses, CapabilityUse{
				Path:    at.Path + ".rules",
				Subject: fmt.Sprintf("decision '%s': the authz signal '%s', which %s,", at.Decision.Name, name, reads),
			})
		}
	}
	for i, provider := range cfg.RateLimit.Providers {
		for _, rule := range provider.Rules {
			if rule.Match.User == "" && rule.Match.Group == "" {
				continue
			}
			uses = append(uses, CapabilityUse{
				Path:    "global.services.ratelimit.providers[" + strconv.Itoa(i) + "].rules[" + rule.Name + "].match",
				Subject: fmt.Sprintf("rate limit rule '%s': match.user and match.group, which %s,", rule.Name, reads),
			})
		}
	}
	if len(cfg.Authz.Providers) > 0 {
		uses = append(uses, CapabilityUse{
			Path:    "global.services.authz.providers",
			Subject: fmt.Sprintf("global.services.authz.providers, which resolve per-user API keys and %s,", reads),
		})
	}
	if cfg.Headers.Debug == DebugHeadersAuthenticated {
		uses = append(uses, CapabilityUse{
			Path:    "global.router.headers.debug",
			Subject: fmt.Sprintf("global.router.headers.debug: %s, which %s,", DebugHeadersAuthenticated, reads),
		})
	}
	return uses
}

// UntrustedIdentityWarning is the startup warning for a standalone Router
// whose features record a user but whose listeners trust no identity source:
// requests stay valid and anonymous, so those features record no user. It is
// empty when there is nothing to warn about.
func UntrustedIdentityWarning(cfg *RouterConfig) string {
	if cfg == nil || cfg.TrustsIdentityHeaders() {
		return ""
	}
	var features []string
	if memoryRecordsUsers(cfg) {
		features = append(features, "memory")
	}
	if cfg.UsesRouterReplay() {
		features = append(features, "router replay")
	}
	if len(features) == 0 {
		return ""
	}
	return fmt.Sprintf("%s record the user of each request, but no listener trusts an identity source, "+
		"so every request is anonymous; behind an authenticating proxy, set listeners[].identity.trust_headers: true",
		strings.Join(features, " and "))
}

func memoryRecordsUsers(cfg *RouterConfig) bool {
	return cfg.UsesMemory()
}

func firstSignalOfType(node *RuleNode, signalType string) (string, bool) {
	if node == nil {
		return "", false
	}
	if node.IsLeaf() {
		return node.Name, strings.EqualFold(node.Type, signalType)
	}
	for i := range node.Conditions {
		if name, ok := firstSignalOfType(&node.Conditions[i], signalType); ok {
			return name, true
		}
	}
	return "", false
}
