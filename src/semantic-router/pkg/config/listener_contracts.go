package config

import (
	"fmt"
	"net/netip"
	"path/filepath"
	"strings"
)

// CapabilityListenerModels is a listener's model allow-list, which only the
// standalone gateway enforces: the Envoy listener the CLI generates does not
// yet, so behind Envoy every model would stay reachable.
const CapabilityListenerModels = "listener_models"

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
	for _, listener := range cfg.Listeners {
		if listener.TLS != nil &&
			(strings.TrimSpace(listener.TLS.CertFile) == "" || strings.TrimSpace(listener.TLS.KeyFile) == "") {
			return fmt.Errorf("listener '%s': tls needs both cert_file and key_file", listener.Name)
		}
		if err := validateListenerModels(listener); err != nil {
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
