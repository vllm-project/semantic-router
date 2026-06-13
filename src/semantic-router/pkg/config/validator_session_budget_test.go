package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("validateSessionTokenBudget", func() {
	It("accepts default zero-value config (disabled)", func() {
		Expect(validateSessionTokenBudget(SessionTokenBudgetConfig{})).To(Succeed())
	})

	It("accepts an enabled budget with no explicit thresholds", func() {
		cfg := SessionTokenBudgetConfig{Enabled: true, BudgetTokens: 40000}
		Expect(validateSessionTokenBudget(cfg)).To(Succeed())
	})

	It("accepts ascending thresholds", func() {
		cfg := SessionTokenBudgetConfig{
			Enabled:      true,
			BudgetTokens: 40000,
			Thresholds:   SessionTokenBudgetThresholds{ShapeTools: 1.0, Compress: 1.5, Downgrade: 2.0, Terminate: 3.0},
		}
		Expect(validateSessionTokenBudget(cfg)).To(Succeed())
	})

	It("rejects a negative budget", func() {
		cfg := SessionTokenBudgetConfig{Enabled: true, BudgetTokens: -1}
		err := validateSessionTokenBudget(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("budget_tokens"))
	})

	It("rejects enabled with a zero budget (silent no-op footgun)", func() {
		cfg := SessionTokenBudgetConfig{Enabled: true, BudgetTokens: 0}
		err := validateSessionTokenBudget(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("budget_tokens"))
	})

	It("allows disabled with a zero budget", func() {
		Expect(validateSessionTokenBudget(SessionTokenBudgetConfig{Enabled: false, BudgetTokens: 0})).To(Succeed())
	})

	It("rejects a negative threshold", func() {
		cfg := SessionTokenBudgetConfig{
			Enabled:      true,
			BudgetTokens: 40000,
			Thresholds:   SessionTokenBudgetThresholds{Compress: -0.5},
		}
		err := validateSessionTokenBudget(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("compress"))
	})

	It("rejects non-ascending thresholds (compress below shape_tools)", func() {
		cfg := SessionTokenBudgetConfig{
			Enabled:      true,
			BudgetTokens: 40000,
			Thresholds:   SessionTokenBudgetThresholds{ShapeTools: 2.0, Compress: 1.5},
		}
		err := validateSessionTokenBudget(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("ascending"))
	})

	It("rejects terminate below downgrade", func() {
		cfg := SessionTokenBudgetConfig{
			Enabled:      true,
			BudgetTokens: 40000,
			Thresholds:   SessionTokenBudgetThresholds{Downgrade: 2.0, Terminate: 1.5},
		}
		err := validateSessionTokenBudget(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("ascending"))
	})
})
