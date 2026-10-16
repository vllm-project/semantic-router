package handlers

import (
	"fmt"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type canonicalConfigTransport struct {
	routerconfig.CanonicalConfig `yaml:",inline"`
	globalOverrideRaw            *yaml.Node
}

// Persist the original override, not the typed effective global: some nested
// bool fields have omitempty tags, so even a resolved explicit false would be
// lost on serialization and replaced by a default true on the next parse.
func (config canonicalConfigTransport) MarshalYAML() (any, error) {
	var document yaml.Node
	if err := document.Encode(config.CanonicalConfig); err != nil {
		return nil, err
	}
	if config.globalOverrideRaw != nil {
		for i := 0; i+1 < len(document.Content); i += 2 {
			if document.Content[i].Value == "global" {
				document.Content[i+1] = config.globalOverrideRaw
				break
			}
		}
	}
	return &document, nil
}

// Full config transports must resolve sparse global overrides before a typed
// round trip emits omitted booleans as false (notably clear_route_cache). Keep
// this at the shared read/decode boundary so setup import, validation, activation,
// and the ordinary JSON editor all preserve the Router's effective defaults.
func resolveTransportGlobal(data []byte, value any) error {
	var canonical *routerconfig.CanonicalConfig
	var rawTarget **yaml.Node
	switch config := value.(type) {
	case *canonicalConfigTransport:
		canonical = &config.CanonicalConfig
		rawTarget = &config.globalOverrideRaw
	case *setupConfigFile:
		canonical = &config.CanonicalConfig
		rawTarget = &config.globalOverrideRaw
	default:
		return nil
	}
	if canonical.Global == nil {
		// An omitted global block must remain omitted in setup patches, where
		// it means to preserve the bootstrap document's existing global block.
		return nil
	}
	var raw struct {
		Global yaml.Node `yaml:"global"`
	}
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return err
	}
	// Resolve aliases before detaching global from the document: an anchor
	// defined in another section will not survive that section's typed export.
	var override any
	if err := raw.Global.Decode(&override); err != nil {
		return err
	}
	if err := raw.Global.Encode(override); err != nil {
		return err
	}
	// Setup patches can omit providers and routing, so normalize just the raw
	// global block with the Router parser. This also applies its module-specific
	// sparse override rules and preserves environment references for persistence.
	globalDocument, err := yaml.Marshal(struct {
		Version string    `yaml:"version"`
		Global  yaml.Node `yaml:"global"`
	}{routerconfig.CanonicalConfigVersion, raw.Global})
	if err != nil {
		return err
	}
	parsed, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(globalDocument)
	if err != nil {
		return fmt.Errorf("invalid global configuration: %w", err)
	}
	canonical.Global = routerconfig.CanonicalGlobalFromRouterConfig(parsed)
	*rawTarget = &raw.Global
	return nil
}
