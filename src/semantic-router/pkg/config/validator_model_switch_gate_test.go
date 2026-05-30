package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
)

var _ = Describe("validateModelSwitchGate", func() {
	It("accepts default zero-value config (empty mode)", func() {
		Expect(validateModelSwitchGate(ModelSwitchGateConfig{})).To(Succeed())
	})

	It("accepts mode=shadow", func() {
		cfg := ModelSwitchGateConfig{Enabled: true, Mode: "shadow"}
		Expect(validateModelSwitchGate(cfg)).To(Succeed())
	})

	It("accepts mode=enforce", func() {
		cfg := ModelSwitchGateConfig{Enabled: true, Mode: "enforce"}
		Expect(validateModelSwitchGate(cfg)).To(Succeed())
	})

	It("rejects misspelled mode", func() {
		cfg := ModelSwitchGateConfig{Enabled: true, Mode: "enforced"}
		err := validateModelSwitchGate(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("model_switch_gate.mode"))
		Expect(err.Error()).To(ContainSubstring("enforced"))
	})

	It("rejects negative min_switch_advantage", func() {
		cfg := ModelSwitchGateConfig{Enabled: true, Mode: "shadow", MinSwitchAdvantage: -0.1}
		err := validateModelSwitchGate(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("min_switch_advantage"))
	})

	It("rejects negative default_handoff_penalty", func() {
		cfg := ModelSwitchGateConfig{Enabled: true, Mode: "shadow", DefaultHandoffPenalty: -0.05}
		err := validateModelSwitchGate(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("default_handoff_penalty"))
	})

	It("rejects negative cache_warmth_weight", func() {
		cfg := ModelSwitchGateConfig{Enabled: true, Mode: "shadow", CacheWarmthWeight: -0.5}
		err := validateModelSwitchGate(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("cache_warmth_weight"))
	})

	It("accepts zero weights and penalties", func() {
		cfg := ModelSwitchGateConfig{
			Enabled:               true,
			Mode:                  "enforce",
			MinSwitchAdvantage:    0,
			DefaultHandoffPenalty: 0,
			CacheWarmthWeight:     0,
		}
		Expect(validateModelSwitchGate(cfg)).To(Succeed())
	})
})

var _ = Describe("validateSessionAwareSelectionConfig", func() {
	It("preserves explicit zero values through YAML unmarshalling", func() {
		var cfg SessionAwareSelectionConfig
		Expect(yaml.Unmarshal([]byte(`
idle_timeout_seconds: 0
switch_margin: 0
prefix_cache_weight: 0
`), &cfg)).To(Succeed())

		Expect(cfg.IdleTimeoutSeconds).NotTo(BeNil())
		Expect(*cfg.IdleTimeoutSeconds).To(Equal(0))
		Expect(cfg.SwitchMargin).NotTo(BeNil())
		Expect(*cfg.SwitchMargin).To(Equal(0.0))
		Expect(cfg.PrefixCacheWeight).NotTo(BeNil())
		Expect(*cfg.PrefixCacheWeight).To(Equal(0.0))
	})

	It("accepts explicit zero values for disableable penalties and biases", func() {
		cfg := SessionAwareSelectionConfig{
			IdleTimeoutSeconds:    ptrInt(0),
			MinTurnsBeforeSwitch:  ptrInt(0),
			SwitchMargin:          ptrFloat64(0),
			StayBias:              ptrFloat64(0),
			ToolLoopStayBias:      ptrFloat64(0),
			PrefixCacheWeight:     ptrFloat64(0),
			HandoffPenaltyWeight:  ptrFloat64(0),
			DefaultHandoffPenalty: ptrFloat64(0),
			SwitchHistoryWeight:   ptrFloat64(0),
		}

		Expect(validateSessionAwareSelectionConfig(cfg)).To(Succeed())
	})

	It("rejects negative optional values", func() {
		cfg := SessionAwareSelectionConfig{
			PrefixCacheWeight: ptrFloat64(-0.1),
		}

		err := validateSessionAwareSelectionConfig(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("prefix_cache_weight"))
	})

	It("rejects non-positive multiplier values when explicitly set", func() {
		cfg := SessionAwareSelectionConfig{
			MaxCacheCostMultiplier: ptrFloat64(0),
		}

		err := validateSessionAwareSelectionConfig(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("max_cache_cost_multiplier"))
		Expect(err.Error()).To(ContainSubstring("> 0"))
	})
})

func ptrInt(v int) *int {
	return &v
}
