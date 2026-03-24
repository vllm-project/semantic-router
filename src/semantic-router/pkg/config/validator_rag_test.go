package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("validateDecisionRAGAndMemoryPlugins", func() {
	It("rejects invalid RAG plugin config on a decision", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{
					{
						Name: "bad-rag",
						ModelRefs: []ModelRef{{
							Model:                 "model-a",
							ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
						}},
						Plugins: []DecisionPlugin{
							{
								Type: "rag",
								Configuration: MustStructuredPayload(map[string]interface{}{
									"enabled": true,
									"backend": "milvus",
								}),
							},
						},
					},
				},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("RAG plugin"))
	})

	It("accepts valid RAG plugin config on a decision", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{
					{
						Name: "good-rag",
						ModelRefs: []ModelRef{{
							Model:                 "model-a",
							ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
						}},
						Plugins: []DecisionPlugin{
							{
								Type: "rag",
								Configuration: MustStructuredPayload(map[string]interface{}{
									"enabled": true,
									"backend": "milvus",
									"backend_config": map[string]interface{}{
										"collection": "my_docs",
									},
								}),
							},
						},
					},
				},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).NotTo(HaveOccurred())
	})
})
