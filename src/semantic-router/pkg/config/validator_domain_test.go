package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestSupportedRoutingDomainNamesStayInSyncWithClassifierMapping(t *testing.T) {
	data, err := os.ReadFile(filepath.Join(referenceConfigRepoRoot(t), "models", "mmbert32k-intent-classifier-merged", "category_mapping.json"))
	if err != nil {
		t.Fatalf("read category_mapping.json: %v", err)
	}

	var mapping struct {
		CategoryToIdx map[string]int `json:"category_to_idx"`
	}
	if err := json.Unmarshal(data, &mapping); err != nil {
		t.Fatalf("unmarshal category mapping: %v", err)
	}

	got := SupportedRoutingDomainNames()
	want := make([]string, 0, len(mapping.CategoryToIdx))
	for name := range mapping.CategoryToIdx {
		want = append(want, name)
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
