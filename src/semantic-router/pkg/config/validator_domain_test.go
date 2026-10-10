package config

import (
	"math"
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

func TestValidateDomainContractsAcceptsValidModelScores(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				Categories: []Category{
					{
						CategoryMetadata: CategoryMetadata{Name: "math"},
						ModelScores: []ModelScore{
							{Model: "model-a", Score: 0.0},
							{Model: "model-b", Score: 0.5},
							{Model: "model-c", Score: 1.0},
						},
					},
				},
			},
		},
	}

	if err := validateDomainContracts(cfg); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestValidateDomainContractsRejectsNonFiniteModelScore(t *testing.T) {
	tests := []struct {
		name  string
		score float64
	}{
		{name: "NaN", score: math.NaN()},
		{name: "+Inf", score: math.Inf(1)},
		{name: "-Inf", score: math.Inf(-1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &RouterConfig{
				IntelligentRouting: IntelligentRouting{
					Signals: Signals{
						Categories: []Category{
							{
								CategoryMetadata: CategoryMetadata{Name: "math"},
								ModelScores: []ModelScore{
									{Model: "model-a", Score: tt.score},
								},
							},
						},
					},
				},
			}

			err := validateDomainContracts(cfg)
			if err == nil {
				t.Fatalf("expected error for non-finite score %v, got nil", tt.score)
			}
			if !strings.Contains(err.Error(), "score must be finite") {
				t.Fatalf("expected error to contain 'score must be finite', got: %v", err)
			}
			if !strings.Contains(err.Error(), "routing.signals.domains") {
				t.Fatalf("expected error to contain 'routing.signals.domains', got: %v", err)
			}
		})
	}
}

func TestValidateDomainContractsRejectsOutOfBandModelScore(t *testing.T) {
	tests := []struct {
		name  string
		score float64
	}{
		{name: "negative slightly below zero", score: -0.01},
		{name: "negative large", score: -100.0},
		{name: "above one slightly", score: 1.01},
		{name: "above one large", score: 50.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &RouterConfig{
				IntelligentRouting: IntelligentRouting{
					Signals: Signals{
						Categories: []Category{
							{
								CategoryMetadata: CategoryMetadata{Name: "code"},
								ModelScores: []ModelScore{
									{Model: "model-b", Score: tt.score},
								},
							},
						},
					},
				},
			}

			err := validateDomainContracts(cfg)
			if err == nil {
				t.Fatalf("expected error for out-of-band score %v, got nil", tt.score)
			}
			if !strings.Contains(err.Error(), "score must be between 0.0 and 1.0") {
				t.Fatalf("expected error to contain 'score must be between 0.0 and 1.0', got: %v", err)
			}
			if !strings.Contains(err.Error(), "routing.signals.domains") {
				t.Fatalf("expected error to contain 'routing.signals.domains', got: %v", err)
			}
		})
	}
}

func TestValidateDomainContractsRejectsNaNFromYAML(t *testing.T) {
	rawYAML := `
categories:
  - name: math
    model_scores:
      - model: gpt-4
        score: .nan
`
	var signals Signals
	if err := yamlv3.Unmarshal([]byte(rawYAML), &signals); err != nil {
		t.Fatalf("unmarshal YAML: %v", err)
	}

	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: signals,
		},
	}

	err := validateDomainContracts(cfg)
	if err == nil {
		t.Fatal("expected error for score: .nan from YAML, got nil")
	}
	if !strings.Contains(err.Error(), "score must be finite") {
		t.Fatalf("expected error to contain 'score must be finite', got: %v", err)
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
