package config

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

func TestSupportedRoutingDomainNamesStayInSyncWithCommittedDomainContract(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(referenceConfigRepoRoot(t), "config", "signal", "domain", "mmlu.yaml"))
	if err != nil {
		t.Fatalf("read config/signal/domain/mmlu.yaml: %v", err)
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

func TestValidateDomainContractsRejectsUnsupportedMMLUCategory(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				Categories: []Category{testDomainCategory("compact", "computer_science")},
			},
		},
	}

	err := validateDomainContracts(cfg)
	if err == nil {
		t.Fatal("expected unsupported mmlu_categories error")
	}
}

func TestValidateDomainContractsRejectsUnsupportedSoftmaxGroupImplicitDomain(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				Categories: []Category{testDomainCategory("balance_demo_compact")},
			},
			Projections: Projections{
				Partitions: []ProjectionPartition{{
					Name:      "domain_partition",
					Semantics: "softmax_exclusive",
					Members:   []string{"balance_demo_compact"},
					Default:   "balance_demo_compact",
				}},
			},
		},
	}

	err := validateDomainContracts(cfg)
	if err == nil {
		t.Fatal("expected unsupported softmax domain member error")
	}
}

func TestValidateDomainContractsRejectsUndeclaredDecisionDomain(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				Categories: []Category{testDomainCategory("math")},
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

	err := validateDomainContracts(cfg)
	if err == nil {
		t.Fatal("expected undeclared decision domain error")
	}
}

func testDomainCategory(name string, mmluCategories ...string) Category {
	return Category{
		CategoryMetadata: CategoryMetadata{
			Name:           name,
			MMLUCategories: mmluCategories,
		},
	}
}
