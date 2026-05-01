package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
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
