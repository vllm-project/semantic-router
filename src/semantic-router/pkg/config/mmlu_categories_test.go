// Copyright 2025 The vLLM Semantic Router Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
