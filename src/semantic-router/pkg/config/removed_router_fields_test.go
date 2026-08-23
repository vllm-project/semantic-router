package config

import (
	"strings"
	"testing"
)

func TestParseYAMLBytesRejectsRemovedSkipProcessing(t *testing.T) {
	manifest := strings.Replace(
		entrypointRulesYAML,
		"  services:\n",
		"  router:\n    skip_processing: {enabled: true}\n  services:\n",
		1,
	)
	_, err := ParseYAMLBytes([]byte(manifest))
	if err == nil || !strings.Contains(err.Error(), "global.router.skip_processing has been removed") {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
}

func TestParseYAMLBytesRejectsRemovedConfigurationAuthorities(t *testing.T) {
	tests := []struct {
		name     string
		fragment string
	}{
		{
			name:     "router config source",
			fragment: "  router:\n    config_source: kubernetes\n",
		},
		{
			name:     "routing manifest indirection",
			fragment: "  control_plane:\n    mode: standalone\n    routing_manifest_file: /tmp/routing.yaml\n",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manifest := strings.Replace(entrypointRulesYAML, "global:\n", "global:\n"+test.fragment, 1)
			if _, err := ParseYAMLBytes([]byte(manifest)); err == nil {
				t.Fatalf("ParseYAMLBytes() accepted removed configuration authority:\n%s", test.fragment)
			}
		})
	}
}
