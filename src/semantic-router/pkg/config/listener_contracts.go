package config

import (
	"fmt"
	"maps"
	"net/netip"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// CapabilityListenerModels is a listener's model allow-list, which only the
// standalone gateway enforces: the Envoy listener the CLI generates does not
// yet, so behind Envoy every model would stay reachable.
const CapabilityListenerModels = "listener_models"

// DefaultSystemOneMaxDeadlineMs is the longest a listener's System One API
// waits for a model when its systemone.max_deadline_ms is omitted: 10 minutes.
const DefaultSystemOneMaxDeadlineMs = 600_000

func init() {
	GatewayCapabilities.MustRegister(CapabilityListenerModels, GatewayCapability{
		Modes: []GatewayMode{GatewayStandalone},
		Uses: func(cfg *RouterConfig) []CapabilityUse {
			var uses []CapabilityUse
			for _, listener := range cfg.Listeners {
				if len(listener.Models) > 0 {
					uses = append(uses, CapabilityUse{
						Path:    "listeners[" + listener.Name + "].models",
						Subject: fmt.Sprintf("listener '%s': models", listener.Name),
					})
				}
			}
			return uses
		},
		Unserved: "is unsupported with --gateway extproc, whose Envoy listener does not enforce it; " +
			"serve with --gateway standalone or remove it",
	})
}

// validateListenerContracts checks the listener settings every gateway mode
// reads the same way.
func validateListenerContracts(cfg *RouterConfig) error {
	names := make(map[string]bool, len(cfg.Listeners))
	for _, listener := range cfg.Listeners {
		if names[listener.Name] {
			return fmt.Errorf("listener '%s' is defined twice; give each listener its own name", listener.Name)
		}
		names[listener.Name] = true
		if listener.TLS != nil &&
			(strings.TrimSpace(listener.TLS.CertFile) == "" || strings.TrimSpace(listener.TLS.KeyFile) == "") {
			return fmt.Errorf("listener '%s': tls needs both cert_file and key_file", listener.Name)
		}
		if err := validateListenerModels(listener); err != nil {
			return err
		}
		if err := validateListenerSystemOne(cfg, listener); err != nil {
			return err
		}
		if listener.Identity == nil {
			continue
		}
		if len(listener.Identity.TrustedPeers) > 0 && !listener.Identity.TrustHeaders {
			return fmt.Errorf("listener '%s': identity.trusted_peers applies only with identity.trust_headers: true", listener.Name)
		}
		if _, err := listener.Identity.PeerPrefixes(); err != nil {
			return fmt.Errorf("listener '%s': %w", listener.Name, err)
		}
	}
	return nil
}

func validateListenerSystemOne(cfg *RouterConfig, listener Listener) error {
	if listener.SystemOne == nil {
		return nil
	}
	if len(listener.SystemOne.Models) == 0 {
		return fmt.Errorf("listener %q: systemone.models must explicitly name at least one public model", listener.Name)
	}
	if limit := listener.SystemOne.MaxDeadlineMs; limit != nil && *limit <= 0 {
		return fmt.Errorf("listener %q: systemone.max_deadline_ms must be greater than zero, got %d", listener.Name, *limit)
	}
	seen := map[string]bool{}
	for _, model := range listener.SystemOne.Models {
		if model == "" || model != strings.TrimSpace(model) || seen[model] {
			return fmt.Errorf("listener %q: systemone.models must contain distinct non-empty model names without surrounding spaces", listener.Name)
		}
		seen[model] = true
		// A routed entrypoint is not one deployment. A provider alias is
		// checked in validateNativeRoutingConfig, published or not.
		if _, routed := cfg.ResolveEntrypoint(SystemOneAPI, model); routed || cfg.IsSystemOneBackend(model) {
			continue
		}
		name, deployment, err := cfg.ResolveSystemOneDeployment(model)
		if err != nil {
			return fmt.Errorf("listener %q: systemone.models: %w", listener.Name, err)
		}
		if err := deployment.ValidateDecisionInput(name); err != nil {
			return fmt.Errorf("listener %q: systemone.models: %w", listener.Name, err)
		}
	}
	return nil
}

// MaxDeadline is the longest the listener's System One API waits for a model
// it calls directly.
func (s *ListenerSystemOne) MaxDeadline() time.Duration {
	if s == nil || s.MaxDeadlineMs == nil {
		return DefaultSystemOneMaxDeadlineMs * time.Millisecond
	}
	return time.Duration(*s.MaxDeadlineMs) * time.Millisecond
}

// IsSystemOneBackend reports whether model is an exact provider alias for the
// native System One contract. Chat aliases and upstream-only identities do not
// become native public names.
func (c *RouterConfig) IsSystemOneBackend(model string) bool {
	if c == nil {
		return false
	}
	backend, ok := c.ModelConfig[model]
	return ok && backend.APIFormat == APIFormatSystemOne
}

// ResolveSystemOneDeployment maps a public inference identity to exactly one
// concrete deployment. Internal keys and attached-runtime served names are not
// public aliases. Virtual recipe resolution belongs to ResolveEntrypoint.
func (c *RouterConfig) ResolveSystemOneDeployment(model string) (string, ModelDeployment, error) {
	var name string
	var deployment ModelDeployment
	if model == "" {
		return "", deployment, fmt.Errorf("a public System One model is required")
	}
	for key, candidate := range c.ModelDeployments {
		if candidate.PublicModelName() != model {
			continue
		}
		if name != "" {
			return "", ModelDeployment{}, fmt.Errorf("public System One model %q names multiple deployments; assign distinct public_name values", model)
		}
		name, deployment = key, candidate
	}
	if name == "" {
		return "", ModelDeployment{}, fmt.Errorf("public System One model %q is not declared", model)
	}
	if deployment.Provider != ModelRuntimeProvider {
		return "", ModelDeployment{}, fmt.Errorf("public System One model %q requires a model_runtime deployment", model)
	}
	return name, deployment, nil
}

// validateListenerModels checks that every allowed model is a distinct name
// a request can send: request models are matched exactly, after trimming.
func validateListenerModels(listener Listener) error {
	seen := make(map[string]bool, len(listener.Models))
	for index, model := range listener.Models {
		if model == "" || strings.TrimSpace(model) != model {
			return fmt.Errorf("listener '%s': models[%d] %q must be a model name without surrounding spaces", listener.Name, index, model)
		}
		if seen[model] {
			return fmt.Errorf("listener '%s': models lists %q twice", listener.Name, model)
		}
		seen[model] = true
	}
	return nil
}

// PeerPrefixes parses TrustedPeers.
func (i ListenerIdentity) PeerPrefixes() ([]netip.Prefix, error) {
	prefixes := make([]netip.Prefix, 0, len(i.TrustedPeers))
	for index, peer := range i.TrustedPeers {
		prefix, err := netip.ParsePrefix(strings.TrimSpace(peer))
		if err != nil {
			return nil, fmt.Errorf("identity.trusted_peers[%d] %q is not a CIDR such as 10.0.0.0/8 or 10.0.0.7/32", index, peer)
		}
		prefixes = append(prefixes, prefix.Masked())
	}
	return prefixes, nil
}

// TrustsIdentityHeaders reports whether a listener keeps client identity
// headers, which identity-based policy needs in standalone mode.
func (c *RouterConfig) TrustsIdentityHeaders() bool {
	for _, listener := range c.Listeners {
		if listener.Identity != nil && listener.Identity.TrustHeaders {
			return true
		}
	}
	return false
}

// Files returns the certificate and key paths, with relative paths resolved
// against baseDir, the configuration's directory.
func (t ListenerTLS) Files(baseDir string) (certFile, keyFile string) {
	resolve := func(path string) string {
		path = strings.TrimSpace(path)
		if filepath.IsAbs(path) || baseDir == "" {
			return filepath.Clean(path)
		}
		return filepath.Join(baseDir, path)
	}
	return resolve(t.CertFile), resolve(t.KeyFile)
}

// ListenerRestart is a listener change a standalone Router applies only by
// restarting: Path names the changed field, Message what changed.
type ListenerRestart struct {
	Path    string
	Message string
}

// ListenerRestarts lists what changes from the bound listeners to next that a
// standalone Router, which binds its listeners at startup, cannot apply in
// place: a new or removed listener, or a listener's address, port, timeout
// (also its connections' idle timeout) or TLS file paths. Keys, model
// allow-lists and identity reload in place.
func ListenerRestarts(bound, next []Listener) []ListenerRestart {
	byName := make(map[string]Listener, len(bound))
	for _, listener := range bound {
		byName[listener.Name] = listener
	}
	var restarts []ListenerRestart
	restart := func(path, message string) {
		restarts = append(restarts, ListenerRestart{Path: path, Message: message})
	}
	for _, listener := range next {
		path := "listeners[" + listener.Name + "]"
		previous, ok := byName[listener.Name]
		delete(byName, listener.Name)
		switch {
		case !ok:
			restart(path, "the listener is new")
		case listener.Address != previous.Address:
			restart(path+".address", fmt.Sprintf("the address changes from %q to %q", previous.Address, listener.Address))
		case listener.Port != previous.Port:
			restart(path+".port", fmt.Sprintf("the port changes from %d to %d", previous.Port, listener.Port))
		case listener.Timeout != previous.Timeout:
			restart(path+".timeout", fmt.Sprintf("the timeout changes from %q to %q", previous.Timeout, listener.Timeout))
		case !sameListenerTLS(listener.TLS, previous.TLS):
			restart(path+".tls", "the TLS file paths change")
		}
	}
	for _, removed := range slices.Sorted(maps.Keys(byName)) {
		restart("listeners["+removed+"]", "the listener is removed")
	}
	return restarts
}

func sameListenerTLS(a, b *ListenerTLS) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}
