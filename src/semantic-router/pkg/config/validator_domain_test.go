package config

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

func TestSupportedRoutingDomainNamesStayInSyncWithCommittedDomainContract(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(referenceConfigRepoRoot(t), "config", "fragments", "signal", "domain", "mmlu.yaml"))
	if err != nil {
		t.Fatalf("read config/fragments/signal/domain/mmlu.yaml: %v", err)
	}

	var fragment struct {
		Routing struct {
			Signals struct {
				Domains []Category `yaml:"domains"`
			} `yaml:"signals"`
		} `yaml:"routing"`
	}
	if err := yamlv3.Unmarshal(data, &fragment); err != nil {
		t.Fatalf("unmarshal committed domain contract: %v", err)
	}

	got := SupportedRoutingDomainNames()
	want := make([]string, 0, len(fragment.Routing.Signals.Domains))
	for _, domain := range fragment.Routing.Signals.Domains {
		want = append(want, domain.Name)
	}
	slices.Sort(got)
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Fatalf("supported routing domains mismatch\nwant: %v\ngot:  %v", want, got)
	}
}

func TestValidateDomainContractsAllowsLooseAliasOutsideSoftmaxGroup(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				Categories: []Category{testDomainCategory("balance_demo_compact")},
			},
		},
	}

	if err := validateDomainContracts(cfg); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestValidateDomainContractsAllowsAliasWithSupportedMMLUCategories(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				Categories: []Category{testDomainCategory("compact", "computer science")},
			},
			Decisions: []Decision{{
				Name: "compact_route",
				Rules: RuleNode{
					Type: SignalTypeDomain,
					Name: "compact",
				},
			}},
		},
	}

	if err := validateDomainContracts(cfg); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestApplicationSignalContractsRejectALabelOutsideTheDomainPreset(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				Categories: []Category{testDomainCategory("compact", "computer_science")},
			},
		},
	}

	err := validateApplicationSignalContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), "computer_science") {
		t.Fatalf("expected a label outside the domain preset to be rejected, got %v", err)
	}
}

func TestProjectionPartitionsRejectAMemberThatIsNoDomainRule(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				Categories: []Category{testDomainCategory("math", "math")},
			},
			Projections: Projections{
				Partitions: []ProjectionPartition{{
					Name:      "domain_partition",
					Semantics: "softmax_exclusive",
					Members:   []string{"math", "science"},
					Default:   "math",
				}},
			},
		},
	}

	if err := validateProjectionPartitions(cfg); err == nil {
		t.Fatal("expected a partition member that names no domain rule to be rejected")
	}
}

func TestDecisionContractsRejectAnUndeclaredDomainRule(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				Categories: []Category{testDomainCategory("math", "math")},
			},
			Decisions: []Decision{{
				Name: "science_route",
				Rules: RuleNode{
					Type: SignalTypeDomain,
					Name: "science",
				},
			}},
		},
	}

	err := validateDecisionContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), `unknown signal "science"`) {
		t.Fatalf("expected undeclared decision domain error, got %v", err)
	}
}

func testDomainCategory(name string, mmluCategories ...string) Category {
	return Category{
		Name:   name,
		Labels: mmluCategories,
	}
}
