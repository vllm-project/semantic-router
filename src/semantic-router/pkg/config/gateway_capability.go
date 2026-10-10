package config

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extension"
)

// GatewayMode is how client traffic reaches the Router, as its -gateway flag
// names it.
type GatewayMode string

const (
	// GatewayStandalone is the Router serving its listeners itself.
	GatewayStandalone GatewayMode = "standalone"
	// GatewayExtProc is an Envoy-based gateway calling the Router over
	// ext_proc.
	GatewayExtProc GatewayMode = "extproc"
)

// ParseGatewayMode parses a gateway mode name.
func ParseGatewayMode(name string) (GatewayMode, error) {
	switch mode := GatewayMode(name); mode {
	case GatewayStandalone, GatewayExtProc:
		return mode, nil
	default:
		return "", fmt.Errorf("unknown gateway mode %q: use %q or %q", name, GatewayStandalone, GatewayExtProc)
	}
}

// GatewayCapability is a feature that only some gateway modes serve.
type GatewayCapability struct {
	// Modes serve the capability. A configuration that uses it fails to load,
	// and a reload that does is rejected, in every other mode.
	Modes []GatewayMode
	// Uses reports where a configuration uses the capability.
	Uses func(cfg *RouterConfig) []CapabilityUse
	// Unserved, when set, replaces the default reason a violation gives in a
	// mode that does not serve the capability, such as a remedy that keeps it.
	Unserved string
}

// CapabilityUse is one place a configuration uses a capability.
type CapabilityUse struct {
	// Path locates the use in the canonical document.
	Path string
	// Subject names the use in messages, such as
	// "decision 'x': reliability.idle_timeout".
	Subject string
}

// CapabilityViolation is a use of a capability that the gateway mode does not
// serve.
type CapabilityViolation struct {
	Capability string
	Path       string
	Message    string
}

// GatewayCapabilities holds the capabilities that depend on the gateway mode,
// by name. Packages register theirs from init functions.
var GatewayCapabilities = extension.NewRegistry[GatewayCapability]("gateway capability")

// CheckGatewayCapabilities lists every use in cfg of a capability that mode
// does not serve, in registration order.
func CheckGatewayCapabilities(cfg *RouterConfig, mode GatewayMode) []CapabilityViolation {
	if !cfg.RoutingEnabled() {
		return nil
	}
	var violations []CapabilityViolation
	for _, entry := range GatewayCapabilities.Entries() {
		if entry.Spec.Uses == nil || slices.Contains(entry.Spec.Modes, mode) {
			continue
		}
		reason := entry.Spec.Unserved
		if reason == "" {
			reason = unservedBy(mode, entry.Spec.Modes)
		}
		for _, use := range entry.Spec.Uses(cfg) {
			violations = append(violations, CapabilityViolation{
				Capability: entry.Type,
				Path:       use.Path,
				Message:    use.Subject + " " + reason,
			})
		}
	}
	return violations
}

// ValidateGatewayCapabilities is CheckGatewayCapabilities as one error.
func ValidateGatewayCapabilities(cfg *RouterConfig, mode GatewayMode) error {
	violations := CheckGatewayCapabilities(cfg, mode)
	if len(violations) == 0 {
		return nil
	}
	messages := make([]string, 0, len(violations))
	for _, violation := range violations {
		messages = append(messages, violation.Message)
	}
	return errors.New(strings.Join(messages, "; "))
}

func unservedBy(mode GatewayMode, modes []GatewayMode) string {
	switch {
	case mode == GatewayExtProc && slices.Contains(modes, GatewayStandalone):
		return "is honored only in standalone mode; serve with --gateway standalone or remove it"
	case mode == GatewayStandalone && slices.Contains(modes, GatewayExtProc):
		return "is not supported in standalone mode; serve with --gateway extproc or remove it"
	default:
		return "is not supported with --gateway " + string(mode) + "; remove it"
	}
}

// RoutingDecisionAt is a routing decision with its path in the canonical
// document.
type RoutingDecisionAt struct {
	Path     string
	Decision *Decision
}

// RoutingDecisionsAt lists the decisions of every recipe with their paths.
func (c *RouterConfig) RoutingDecisionsAt() []RoutingDecisionAt {
	if c == nil {
		return nil
	}
	recipes := c.Recipes
	if len(recipes) == 0 {
		recipes = []RoutingRecipe{*c.DefaultRecipe()}
	}
	var decisions []RoutingDecisionAt
	for i := range recipes {
		base := "routing"
		if recipes[i].Name != DefaultRecipeName {
			base = "recipes[" + string(recipes[i].Name) + "].routing"
		}
		for j := range recipes[i].Profile.Decisions {
			decision := &recipes[i].Profile.Decisions[j]
			decisions = append(decisions, RoutingDecisionAt{
				Path: base + ".decisions[" + decision.Name + "]", Decision: decision,
			})
		}
	}
	return decisions
}
