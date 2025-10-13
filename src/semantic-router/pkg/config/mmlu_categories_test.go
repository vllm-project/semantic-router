package config_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("MMLU categories in config YAML", func() {
	It("should unmarshal mmlu_categories into Category struct", func() {
		yamlContent := `
categories:
  - name: "tech"
    mmlu_categories: ["computer science", "engineering"]
    model_scores:
      - model: "phi4"
        score: 0.9
        use_reasoning: false
  - name: "finance"
    mmlu_categories: ["economics"]
    model_scores:
      - model: "gemma3:27b"
        score: 0.8
        use_reasoning: true
  - name: "politics"
    model_scores:
      - model: "gemma3:27b"
        score: 0.6
        use_reasoning: false
`

		var cfg config.RouterConfig
		Expect(yaml.Unmarshal([]byte(yamlContent), &cfg)).To(Succeed())

		Expect(cfg.Categories).To(HaveLen(3))

		Expect(cfg.Categories[0].Name).To(Equal("tech"))
		Expect(cfg.Categories[0].MMLUCategories).To(ConsistOf("computer science", "engineering"))
		Expect(cfg.Categories[0].ModelScores).ToNot(BeEmpty())

		Expect(cfg.Categories[1].Name).To(Equal("finance"))
		Expect(cfg.Categories[1].MMLUCategories).To(ConsistOf("economics"))

		Expect(cfg.Categories[2].Name).To(Equal("politics"))
		Expect(cfg.Categories[2].MMLUCategories).To(BeEmpty())
	})
})
