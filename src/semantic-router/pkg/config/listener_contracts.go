package config

import (
	"fmt"
	"net/netip"
	"path/filepath"
	"strings"
)

// validateListenerContracts checks the listener settings every gateway mode
// reads the same way.
func validateListenerContracts(cfg *RouterConfig) error {
	for _, listener := range cfg.Listeners {
		if listener.TLS != nil &&
			(strings.TrimSpace(listener.TLS.CertFile) == "" || strings.TrimSpace(listener.TLS.KeyFile) == "") {
			return fmt.Errorf("listener '%s': tls needs both cert_file and key_file", listener.Name)
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
