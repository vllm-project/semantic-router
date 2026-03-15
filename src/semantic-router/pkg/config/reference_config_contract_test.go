package config

import (
	"bytes"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

func TestReferenceConfigUsesStrictCanonicalSchema(t *testing.T) {
	data := readReferenceConfigYAML(t)

	decoder := yamlv3.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)

	var canonical CanonicalConfig
	if err := decoder.Decode(&canonical); err != nil {
		t.Fatalf("config/config.yaml no longer matches the strict canonical schema: %v", err)
	}

	if canonical.Version != "v0.3" {
		t.Fatalf("expected reference config version v0.3, got %q", canonical.Version)
	}

	if _, err := ParseYAMLBytes(data); err != nil {
		t.Fatalf("config/config.yaml failed runtime parse validation: %v", err)
	}
}

func TestReferenceConfigCoversCanonicalPublicSurface(t *testing.T) {
	root := loadReferenceConfigRaw(t)

	assertReferenceConfigTopLevelCoverage(t, root)
	assertReferenceConfigProviderCoverage(t, root)
	assertReferenceConfigRoutingCoverage(t, root)
	assertReferenceConfigGlobalCoverage(t, root)
}

func TestReferenceConfigCoversSupportedRoutingSurfaces(t *testing.T) {
	root := loadReferenceConfigRaw(t)
	decisions := mustSliceAt(t, root, "routing", "decisions")

	assertSupportedSignalTypesInReferenceConfig(t, root)
	assertReferenceLoRACatalogCoverage(t, root)
	assertSupportedAlgorithmsInReferenceConfig(t, decisions)
	assertSupportedPluginsInReferenceConfig(t, decisions)
	assertDecisionRuleCompositionInReferenceConfig(t, decisions)
}
