package config

import (
	"fmt"
	"path/filepath"
	"slices"
)

// CapabilityListenerKeyFiles is a listener's api_key_files, which only the
// standalone gateway reads: the Envoy listener the CLI generates holds its
// keys in its own configuration.
const CapabilityListenerKeyFiles = "listener_key_files"

func init() {
	GatewayCapabilities.MustRegister(CapabilityListenerKeyFiles, GatewayCapability{
		Modes: []GatewayMode{GatewayStandalone},
		Uses: func(cfg *RouterConfig) []CapabilityUse {
			var uses []CapabilityUse
			for _, listener := range cfg.Listeners {
				if len(listener.APIKeyFiles) > 0 {
					uses = append(uses, CapabilityUse{
						Path:    listenerPath(listener, "api_key_files"),
						Subject: fmt.Sprintf("listener '%s': api_key_files", listener.Name),
					})
				}
			}
			return uses
		},
		Unserved: "is read only in standalone mode; with --gateway extproc write the keys in api_keys, " +
			`as "${NAME}" references to the environment`,
	})
}

// ClientKeys are the keys a listener accepts: its api_keys and the keys its
// api_key_files held when the configuration loaded.
func (l Listener) ClientKeys() []string {
	return append(slices.Clone(l.APIKeys), l.FileKeys...)
}

// ResolveClientKeys reads a listener of a document as written the way the
// Router does when it loads the document: api_keys with their environment
// references expanded, then the keys of api_key_files. It serves a reader of
// the document that does not load it, such as the Dashboard's Try it.
func (l Listener) ResolveClientKeys() ([]string, error) {
	keys := make([]string, 0, len(l.APIKeys)+len(l.APIKeyFiles))
	for _, key := range l.APIKeys {
		keys = append(keys, processEnv.expandString(key))
	}
	resolved := []Listener{l}
	if err := readListenerKeyFiles(resolved, false); err != nil {
		return nil, err
	}
	return append(keys, resolved[0].FileKeys...), nil
}

// readListenerKeyFiles reads each listener's api_key_files, at every load so
// a reload picks up a rotated file. An error names the file, never the key.
// With checkOnly it checks the paths and reads no file, as a validation that
// reads no external asset does: the files may be mounted only where the
// Router serves the configuration.
func readListenerKeyFiles(listeners []Listener, checkOnly bool) error {
	for index := range listeners {
		listener := &listeners[index]
		listener.FileKeys = nil
		for _, path := range listener.APIKeyFiles {
			if !filepath.IsAbs(path) {
				return fmt.Errorf("listener '%s': api_key_files must name absolute paths", listener.Name)
			}
			if checkOnly {
				continue
			}
			key, err := readKeyFile("api_key_files", path)
			if err != nil {
				return fmt.Errorf("listener '%s': %w", listener.Name, err)
			}
			listener.FileKeys = append(listener.FileKeys, key)
		}
	}
	return nil
}
