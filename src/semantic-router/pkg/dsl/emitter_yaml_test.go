package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEmitUserYAMLUsesProjectionsWithoutLegacySignalGroupsKey(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Projections: config.Projections{
				Partitions: []config.ProjectionPartition{
					{
						Name:      "domain_partition",
						Semantics: "exclusive",
						Members:   []string{"math", "general"},
						Default:   "general",
					},
				},
			},
		},
	}

	userYAML, err := EmitUserYAML(cfg)
	if err != nil {
		t.Fatalf("EmitUserYAML error: %v", err)
	}

	yamlStr := string(userYAML)
	if strings.Contains(yamlStr, "signal_groups:") {
		t.Fatalf("legacy signal_groups key should not be emitted:\n%s", yamlStr)
	}
	if !strings.Contains(yamlStr, "projections:") || !strings.Contains(yamlStr, "partitions:") {
		t.Fatalf("expected projections.partitions in user YAML:\n%s", yamlStr)
	}
	if !strings.Contains(yamlStr, "name: domain_partition") {
		t.Fatalf("expected partition name in user YAML:\n%s", yamlStr)
	}
}
