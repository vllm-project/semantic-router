package config

import (
	"os"
	"path/filepath"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("normalizeLegacyLatencyRouting", func() {
	It("migrates old-only latency condition to latency_aware algorithm", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Signals: Signals{
					LatencyRules: []LatencyRule{
						{
							Name:           "low_latency",
							TPOTPercentile: 20,
							TTFTPercentile: 30,
							Description:    "legacy rule",
						},
					},
				},
				Decisions: []Decision{
					{
						Name: "math-route",
						Rules: RuleCombination{
							Operator: "AND",
							Conditions: []RuleCondition{
								{Type: "domain", Name: "mathematics"},
								{Type: "latency", Name: "low_latency"},
							},
						},
					},
				},
			},
		}

		err := normalizeLegacyLatencyRouting(cfg)
		Expect(err).NotTo(HaveOccurred())

		Expect(cfg.Decisions).To(HaveLen(1))
		Expect(cfg.Decisions[0].Rules.Conditions).To(HaveLen(1))
		Expect(cfg.Decisions[0].Rules.Conditions[0]).To(Equal(RuleCondition{Type: "domain", Name: "mathematics"}))
		Expect(cfg.Decisions[0].Algorithm).NotTo(BeNil())
		Expect(cfg.Decisions[0].Algorithm.Type).To(Equal("latency_aware"))
		Expect(cfg.Decisions[0].Algorithm.LatencyAware).NotTo(BeNil())
		Expect(cfg.Decisions[0].Algorithm.LatencyAware.TPOTPercentile).To(Equal(20))
		Expect(cfg.Decisions[0].Algorithm.LatencyAware.TTFTPercentile).To(Equal(30))
		Expect(cfg.Decisions[0].Algorithm.LatencyAware.Description).To(Equal("legacy rule"))
		Expect(cfg.Signals.LatencyRules).To(BeEmpty())
	})

	It("returns error when legacy and latency_aware are mixed", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Signals: Signals{
					LatencyRules: []LatencyRule{{Name: "low_latency", TPOTPercentile: 20}},
				},
				Decisions: []Decision{
					{
						Name: "legacy",
						Rules: RuleCombination{
							Operator: "AND",
							Conditions: []RuleCondition{
								{Type: "latency", Name: "low_latency"},
							},
						},
					},
					{
						Name: "new",
						Rules: RuleCombination{
							Operator: "AND",
							Conditions: []RuleCondition{
								{Type: "domain", Name: "math"},
							},
						},
						Algorithm: &AlgorithmConfig{
							Type:         "latency_aware",
							LatencyAware: &LatencyAwareAlgorithmConfig{TPOTPercentile: 20},
						},
					},
				},
			},
		}

		err := normalizeLegacyLatencyRouting(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("cannot be used with decision.algorithm.type=latency_aware"))
	})

	It("returns error when latency condition references unknown legacy rule", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Signals: Signals{
					LatencyRules: []LatencyRule{{Name: "known_rule", TPOTPercentile: 20}},
				},
				Decisions: []Decision{
					{
						Name: "legacy",
						Rules: RuleCombination{
							Operator: "AND",
							Conditions: []RuleCondition{
								{Type: "domain", Name: "math"},
								{Type: "latency", Name: "unknown_rule"},
							},
						},
					},
				},
			},
		}

		err := normalizeLegacyLatencyRouting(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("references unknown latency rule"))
	})

	It("returns error when a decision has multiple legacy latency conditions", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Signals: Signals{
					LatencyRules: []LatencyRule{
						{Name: "low_latency", TPOTPercentile: 20},
						{Name: "ultra_low_latency", TPOTPercentile: 10},
					},
				},
				Decisions: []Decision{
					{
						Name: "legacy",
						Rules: RuleCombination{
							Operator: "AND",
							Conditions: []RuleCondition{
								{Type: "domain", Name: "math"},
								{Type: "latency", Name: "low_latency"},
								{Type: "latency", Name: "ultra_low_latency"},
							},
						},
					},
				},
			},
		}

		err := normalizeLegacyLatencyRouting(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("multiple legacy latency conditions are not supported"))
	})

	It("returns error when legacy latency condition uses OR operator", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Signals: Signals{
					LatencyRules: []LatencyRule{{Name: "low_latency", TPOTPercentile: 20}},
				},
				Decisions: []Decision{
					{
						Name: "legacy",
						Rules: RuleCombination{
							Operator: "OR",
							Conditions: []RuleCondition{
								{Type: "domain", Name: "math"},
								{Type: "latency", Name: "low_latency"},
							},
						},
					},
				},
			},
		}

		err := normalizeLegacyLatencyRouting(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("only AND is supported"))
	})

	It("returns error when removing latency condition leaves no conditions", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Signals: Signals{
					LatencyRules: []LatencyRule{{Name: "low_latency", TPOTPercentile: 20}},
				},
				Decisions: []Decision{
					{
						Name: "legacy",
						Rules: RuleCombination{
							Operator: "AND",
							Conditions: []RuleCondition{
								{Type: "latency", Name: "low_latency"},
							},
						},
					},
				},
			},
		}

		err := normalizeLegacyLatencyRouting(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("no non-latency conditions remain"))
	})

	It("returns error when decision already has algorithm configured", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Signals: Signals{
					LatencyRules: []LatencyRule{{Name: "low_latency", TPOTPercentile: 20}},
				},
				Decisions: []Decision{
					{
						Name: "legacy",
						Rules: RuleCombination{
							Operator: "AND",
							Conditions: []RuleCondition{
								{Type: "domain", Name: "math"},
								{Type: "latency", Name: "low_latency"},
							},
						},
						Algorithm: &AlgorithmConfig{Type: "static"},
					},
				},
			},
		}

		err := normalizeLegacyLatencyRouting(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("cannot be auto-migrated when decision.algorithm is configured"))
	})

	It("migrates multiple decisions when each has a valid single legacy latency condition", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Signals: Signals{
					LatencyRules: []LatencyRule{
						{Name: "low_latency", TPOTPercentile: 20, TTFTPercentile: 30},
						{Name: "strict_latency", TPOTPercentile: 10, TTFTPercentile: 15},
					},
				},
				Decisions: []Decision{
					{
						Name: "decision-a",
						Rules: RuleCombination{
							Operator: "AND",
							Conditions: []RuleCondition{
								{Type: "domain", Name: "math"},
								{Type: "latency", Name: "low_latency"},
							},
						},
					},
					{
						Name: "decision-b",
						Rules: RuleCombination{
							Operator: "AND",
							Conditions: []RuleCondition{
								{Type: "keyword", Name: "coding"},
								{Type: "latency", Name: "strict_latency"},
							},
						},
					},
				},
			},
		}

		err := normalizeLegacyLatencyRouting(cfg)
		Expect(err).NotTo(HaveOccurred())
		Expect(cfg.Signals.LatencyRules).To(BeEmpty())

		Expect(cfg.Decisions).To(HaveLen(2))
		Expect(cfg.Decisions[0].Algorithm).NotTo(BeNil())
		Expect(cfg.Decisions[0].Algorithm.Type).To(Equal("latency_aware"))
		Expect(cfg.Decisions[0].Algorithm.LatencyAware).NotTo(BeNil())
		Expect(cfg.Decisions[0].Algorithm.LatencyAware.TPOTPercentile).To(Equal(20))
		Expect(cfg.Decisions[0].Algorithm.LatencyAware.TTFTPercentile).To(Equal(30))
		Expect(cfg.Decisions[0].Rules.Conditions).To(Equal([]RuleCondition{
			{Type: "domain", Name: "math"},
		}))

		Expect(cfg.Decisions[1].Algorithm).NotTo(BeNil())
		Expect(cfg.Decisions[1].Algorithm.Type).To(Equal("latency_aware"))
		Expect(cfg.Decisions[1].Algorithm.LatencyAware).NotTo(BeNil())
		Expect(cfg.Decisions[1].Algorithm.LatencyAware.TPOTPercentile).To(Equal(10))
		Expect(cfg.Decisions[1].Algorithm.LatencyAware.TTFTPercentile).To(Equal(15))
		Expect(cfg.Decisions[1].Rules.Conditions).To(Equal([]RuleCondition{
			{Type: "keyword", Name: "coding"},
		}))
	})
})

var _ = Describe("Parse with legacy latency migration", func() {
	It("applies latency migration before validation", func() {
		tempDir, err := os.MkdirTemp("", "latency_migration_parse_test")
		Expect(err).NotTo(HaveOccurred())
		defer os.RemoveAll(tempDir)

		configPath := filepath.Join(tempDir, "config.yaml")
		content := `
latency_rules:
  - name: low_latency
    tpot_percentile: 20
    ttft_percentile: 30
decisions:
  - name: math_route
    rules:
      operator: AND
      conditions:
        - type: domain
          name: mathematics
        - type: latency
          name: low_latency
    modelRefs:
      - model: model-a
        use_reasoning: true
`
		Expect(os.WriteFile(configPath, []byte(content), 0o644)).To(Succeed())

		cfg, err := Parse(configPath)
		Expect(err).NotTo(HaveOccurred())
		Expect(cfg).NotTo(BeNil())
		Expect(cfg.Decisions).To(HaveLen(1))
		Expect(cfg.Decisions[0].Algorithm).NotTo(BeNil())
		Expect(cfg.Decisions[0].Algorithm.Type).To(Equal("latency_aware"))
		Expect(cfg.Decisions[0].Algorithm.LatencyAware).NotTo(BeNil())
		Expect(cfg.Decisions[0].Algorithm.LatencyAware.TPOTPercentile).To(Equal(20))
		Expect(cfg.Decisions[0].Algorithm.LatencyAware.TTFTPercentile).To(Equal(30))
		Expect(cfg.Decisions[0].Rules.Conditions).To(HaveLen(1))
		Expect(cfg.Decisions[0].Rules.Conditions[0]).To(Equal(RuleCondition{Type: "domain", Name: "mathematics"}))
		Expect(cfg.Signals.LatencyRules).To(BeEmpty())
	})
})
