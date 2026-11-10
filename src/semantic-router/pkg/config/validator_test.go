package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func boolPtr(b bool) *bool { return &b }

var _ = Describe("validateLoRAName", func() {
	loraConfig := func(model string, loras ...string) *RouterConfig {
		adapters := make([]LoRAAdapter, len(loras))
		for i, n := range loras {
			adapters[i] = LoRAAdapter{Name: n}
		}
		return &RouterConfig{
			BackendModels: BackendModels{
				ModelConfig: map[string]ModelParams{
					model: {LoRAs: adapters},
				},
			},
		}
	}

	It("accepts known adapter", func() {
		cfg := loraConfig("qwen3", "sql-expert", "code-review")
		Expect(validateLoRAName(cfg, "qwen3", "sql-expert")).To(Succeed())
		Expect(validateLoRAName(cfg, "qwen3", "code-review")).To(Succeed())
	})

	It("rejects model not in config", func() {
		cfg := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{}}}
		err := validateLoRAName(cfg, "ghost", "adapter")
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("is not declared in providers.models"))
	})

	It("rejects model with no loras", func() {
		cfg := loraConfig("qwen3") // zero adapters
		err := validateLoRAName(cfg, "qwen3", "adapter")
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("declares no providers.models[].loras"))
	})

	It("rejects unknown name and lists available", func() {
		cfg := loraConfig("qwen3", "sql-expert", "code-review")
		err := validateLoRAName(cfg, "qwen3", "nope")
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("sql-expert"))
		Expect(err.Error()).To(ContainSubstring("code-review"))
	})
})

func registerValidateConfigStructureCoreSpecs() {
	registerValidateConfigStructureCoreDispatchSpecs()
	registerValidateConfigStructureOutputContractSpecs()
	registerValidateConfigStructureModelRefSpecs()
}

func registerValidateConfigStructureCoreDispatchSpecs() {
	It("defers routing contracts until CRDs are available in k8s mode", func() {
		cfg := &RouterConfig{
			ConfigSource: ConfigSourceKubernetes,
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{Name: "bad", ModelRefs: nil}},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})

	It("validates shared family contracts after k8s CRD conversion", func() {
		cfg := &RouterConfig{
			ConfigSource: ConfigSourceKubernetes,
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "bad",
					ModelRefs: []ModelRef{{
						Model:                 "",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(false)},
					}},
				}},
			},
		}

		err := ValidateKubernetesConfigContracts(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("model name cannot be empty"))
	})

	It("keeps the shared dispatch table wired for file and k8s validation", func() {
		for _, validators := range [][]configContractValidator{
			globalConfigContractValidators,
			routingConfigContractValidators,
			routingProfileContractValidators,
		} {
			Expect(validators).NotTo(BeEmpty())
			for _, validator := range validators {
				Expect(validator).NotTo(BeNil())
			}
		}
	})

	It("accepts empty config", func() {
		Expect(validateConfigStructure(&RouterConfig{})).To(Succeed())
	})

	It("accepts valid decision", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "ok",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
				}},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})
}

func registerValidateConfigStructureOutputContractSpecs() {
	registerValidateConfigStructureOutputContractAcceptSpecs()
	registerValidateConfigStructureOutputContractRejectSpecs()
}

func registerValidateConfigStructureOutputContractAcceptSpecs() {
	It("accepts typed choice output contract spec", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "choice",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					OutputContractSpec: &OutputContractSpec{
						Type: OutputContractTypeChoice,
						ChoiceSet: &OutputContractChoiceSetSpec{
							Values: []string{"A", "B", "C", "D"},
						},
						Render: &OutputContractRenderSpec{Mode: OutputContractRenderModeValue},
					},
				}},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})

	It("accepts typed terminal action output contract spec", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "json-action",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					OutputContractSpec: &OutputContractSpec{
						Type: OutputContractTypeStructuredJSON,
						JSONSchema: &OutputContractJSONSchemaSpec{
							SchemaRef: OutputContractJSONTerminalActionV1,
						},
						Extract: &OutputContractExtractSpec{
							Mode:    OutputContractExtractModeJSONObject,
							Sources: []string{OutputContractExtractSourceContent, OutputContractExtractSourceCandidateResponses},
						},
					},
				}},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})

	It("accepts typed reference selection output contract spec", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "reference-selection",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					OutputContractSpec: &OutputContractSpec{
						Type: OutputContractTypeReferenceSelect,
						Reference: &OutputContractReferenceSpec{
							Source:   OutputContractExtractSourceCandidateResponses,
							IDFormat: OutputContractReferenceIDFormatIndex,
						},
						Extract: &OutputContractExtractSpec{
							Mode:    OutputContractExtractModeExact,
							Sources: []string{OutputContractExtractSourceContent},
						},
						Postprocess: []OutputContractPostprocess{{
							Type: OutputContractPostprocessDereferenceSelectedReference,
						}},
					},
				}},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})
}

func registerValidateConfigStructureOutputContractRejectSpecs() {
	It("rejects dereference postprocess without reference selection type", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "reference-postprocess",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					OutputContractSpec: &OutputContractSpec{
						Postprocess: []OutputContractPostprocess{{
							Type: OutputContractPostprocessDereferenceSelectedReference,
						}},
					},
				}},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("requires type=reference_selection"))
	})

	It("rejects typed choice output contract without choices", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "bad-choice",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					OutputContractSpec: &OutputContractSpec{Type: OutputContractTypeChoice},
				}},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("choice type requires choice_set.values"))
	})

	It("rejects json object extraction on choice output contract", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "bad-choice-extract",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					OutputContractSpec: &OutputContractSpec{
						Type: OutputContractTypeChoice,
						ChoiceSet: &OutputContractChoiceSetSpec{
							Values: []string{"A", "B", "C", "D"},
						},
						Extract: &OutputContractExtractSpec{Mode: OutputContractExtractModeJSONObject},
					},
				}},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("only supported for structured_json"))
	})

	It("rejects unsupported structured_json schema refs", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "bad-json",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					OutputContractSpec: &OutputContractSpec{
						Type: OutputContractTypeStructuredJSON,
						JSONSchema: &OutputContractJSONSchemaSpec{
							SchemaRef: "custom_v1",
						},
					},
				}},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("unsupported json_schema.schema_ref"))
	})
}

func registerValidateConfigStructureModelRefSpecs() {
	It("accepts empty modelRefs", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{Name: "x", ModelRefs: []ModelRef{}}},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})

	It("rejects blank model name", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "x",
					ModelRefs: []ModelRef{{
						Model:                 "",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(false)},
					}},
				}},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("model name cannot be empty"))
	})

	It("rejects nil use_reasoning", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name:      "x",
					ModelRefs: []ModelRef{{Model: "model-a"}},
				}},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("missing required field 'use_reasoning'"))
	})
}

func registerValidateConfigStructureLoRASpecs() {
	It("validates lora ref in decision", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "x",
					ModelRefs: []ModelRef{{
						Model:                 "qwen3",
						LoRAName:              "sql-expert",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
				}},
			},
			BackendModels: BackendModels{
				ModelConfig: map[string]ModelParams{
					"qwen3": {LoRAs: []LoRAAdapter{{Name: "sql-expert"}}},
				},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})

	It("rejects bad lora ref", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "x",
					ModelRefs: []ModelRef{{
						Model:                 "qwen3",
						LoRAName:              "nope",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
				}},
			},
			BackendModels: BackendModels{
				ModelConfig: map[string]ModelParams{
					"qwen3": {LoRAs: []LoRAAdapter{{Name: "sql-expert"}}},
				},
			},
		}
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("is not declared in providers.models"))
	})
}

func registerValidateConfigStructureAlgorithmSpecs() {
	registerValidateConfigStructureAlgorithmSchemaSpecs()
	registerValidateConfigStructureReMoMSpecs()
	registerValidateConfigStructureFusionSpecs()
	registerValidateConfigStructureWorkflowsSpecs()
	registerValidateConfigStructureAlgorithmTypeMismatchSpecs()
	registerValidateConfigStructureSignalReferenceSpecs()
}

func escalateDecisionConfig(algorithm *AlgorithmConfig) *RouterConfig {
	return &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Decisions: []Decision{{
				Name: "escalate",
				ModelRefs: []ModelRef{
					{Model: "model-a", ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(false)}},
					{Model: "model-b", ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(false)}},
				},
				Algorithm: algorithm,
			}},
		},
	}
}

func registerValidateConfigStructureAlgorithmSchemaSpecs() {
	It("rejects escalate without algorithm.escalate", func() {
		err := validateConfigStructure(escalateDecisionConfig(&AlgorithmConfig{Type: "escalate"}))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("algorithm.type=escalate requires algorithm.escalate configuration"))
	})

	It("accepts a Chat escalate judged by confidence", func() {
		gte := 0.7
		cfg := escalateDecisionConfig(&AlgorithmConfig{
			Type: "escalate", OnError: EscalateOnErrorFail, Budget: &AlgorithmBudget{Deadline: "20s", MaxCalls: 2},
			Escalate: &EscalateAlgorithmConfig{
				Order:  EscalateOrderCost,
				Accept: &EscalateAccept{Confidence: &EscalateConfidence{Method: ConfidenceMethodMargin, GTE: &gte}},
			},
		})
		Expect(validateConfigStructure(withDecisionBackends(cfg))).To(Succeed())
	})

	It("rejects a Chat escalate whose accept is not exactly one check", func() {
		gte := 0.7
		for _, accept := range []*EscalateAccept{
			nil,
			{},
			{Confidence: &EscalateConfidence{GTE: &gte}, Refusal: boolPtr(false)},
			{Refusal: boolPtr(true)},
			{Confidence: &EscalateConfidence{}},
			{Native: []NativeAcceptanceRule{{QuestionType: "choice", Field: "confidence", NumericPredicate: NumericPredicate{GTE: &gte}}}},
		} {
			cfg := escalateDecisionConfig(&AlgorithmConfig{Type: "escalate", Escalate: &EscalateAlgorithmConfig{Accept: accept}})
			Expect(validateConfigStructure(withDecisionBackends(cfg))).To(HaveOccurred(), "accept %+v", accept)
		}
	})

	It("rejects Chat escalate settings that belong to another path or executor", func() {
		gte := 0.7
		accept := &EscalateAccept{Confidence: &EscalateConfidence{GTE: &gte}}
		for _, algorithm := range []*AlgorithmConfig{
			{Type: "escalate", OnError: "skip", Escalate: &EscalateAlgorithmConfig{Accept: accept}},
			{Type: "escalate", Escalate: &EscalateAlgorithmConfig{Order: "automix", Accept: accept}},
			{Type: "escalate", Escalate: &EscalateAlgorithmConfig{Accept: accept, Quality: &EscalateQuality{Type: EscalateQualityUncalibrated}}},
			{Type: "escalate", Escalate: &EscalateAlgorithmConfig{Accept: accept, Stages: []EscalateStage{{Model: "model-a"}}}},
		} {
			Expect(validateConfigStructure(withDecisionBackends(escalateDecisionConfig(algorithm)))).To(HaveOccurred(), "algorithm %+v", algorithm)
		}
	})

	It("rejects multiple algorithm config blocks in one decision", func() {
		cfg := escalateDecisionConfig(&AlgorithmConfig{
			Type:        "multi_factor",
			MultiFactor: &MultiFactorSelectionConfig{},
			Fusion:      &FusionAlgorithmConfig{},
		})
		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("cannot be combined with multiple algorithm config blocks"))
	})
}

func registerValidateConfigStructureReMoMSpecs() {
	registerValidateConfigStructureReMoMRuntimeSpecs()
	registerValidateConfigStructureReMoMDecisionSpecs()
}

func registerValidateConfigStructureReMoMRuntimeSpecs() {
	It("requires explicit entrypoints for orchestration names", func() {
		cfg := &RouterConfig{}
		for _, name := range []string{DefaultReMoMModelName, DefaultFusionModelName, DefaultFlowModelName, OpenRouterFusionModelAlias} {
			Expect(cfg.IsEntrypointModelName(name)).To(BeFalse())
		}
	})
}

func registerValidateConfigStructureReMoMDecisionSpecs() {
	It("accepts remom round_robin distribution", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "remom-round-robin",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type:    "remom",
						OnError: ReMoMOnErrorSkip,
						ReMoM: &ReMoMAlgorithmConfig{
							BreadthSchedule:    []int{3, 2},
							ModelDistribution:  ReMoMDistributionRoundRobin,
							CompactionStrategy: ReMoMCompactionLastNTokens,
						},
					},
				}},
			},
		}

		Expect(validateConfigStructure(withDecisionBackends(cfg))).To(Succeed())
	})

	It("rejects invalid remom model_distribution", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "remom-invalid-distribution",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type: "remom",
						ReMoM: &ReMoMAlgorithmConfig{
							BreadthSchedule:   []int{3, 2},
							ModelDistribution: "uniform",
						},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("algorithm.remom: model_distribution must be one of"))
	})

	It("rejects invalid remom quorum and timeout controls", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "remom-invalid-timeout",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type: "remom",
						ReMoM: &ReMoMAlgorithmConfig{
							BreadthSchedule:        []int{3},
							RoundTimeoutSeconds:    -1,
							MinSuccessfulResponses: 2,
						},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("algorithm.remom: round_timeout_seconds must be >= 1 when set"))
	})

	It("rejects remom synthesis_model outside modelRefs", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "remom-invalid-synthesis-model",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type: "remom",
						ReMoM: &ReMoMAlgorithmConfig{
							BreadthSchedule: []int{3},
							SynthesisModel:  "model-b",
						},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("algorithm.remom: synthesis_model \"model-b\" must reference a decision modelRef"))
	})
}

func registerValidateConfigStructureFusionSpecs() {
	It("allows orchestration aliases only through explicit entrypoints", func() {
		cfg := &RouterConfig{Entrypoints: []EntrypointMapping{{ModelNames: []string{DefaultFusionModelName, OpenRouterFusionModelAlias}, Recipe: DefaultRecipeName}}}
		Expect(cfg.IsEntrypointModelName(DefaultFusionModelName)).To(BeTrue())
		Expect(cfg.IsEntrypointModelName(OpenRouterFusionModelAlias)).To(BeTrue())
	})

	It("accepts fusion with decision modelRefs and no fusion block", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "fusion-with-model-refs",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{Type: "fusion"},
				}},
			},
		}

		Expect(validateConfigStructure(withDecisionBackends(cfg))).To(Succeed())
	})

	It("rejects invalid decision fusion on_error", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "fusion-invalid-on-error",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type:    "fusion",
						OnError: "ignore",
						Fusion:  &FusionAlgorithmConfig{},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring(`algorithm.on_error of fusion must be "skip" or "fail"`))
	})

	It("rejects executor settings on a selector", func() {
		for _, algorithm := range []*AlgorithmConfig{
			{Type: "static", OnError: "skip"},
			{Type: "multi_factor", Budget: &AlgorithmBudget{Deadline: "2s", MaxCalls: 1}, MultiFactor: &MultiFactorSelectionConfig{}},
		} {
			cfg := &RouterConfig{
				IntelligentRouting: IntelligentRouting{
					Decisions: []Decision{{
						Name: "selector-with-executor-settings",
						ModelRefs: []ModelRef{
							{Model: "model-a", ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(false)}},
							{Model: "model-b", ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(false)}},
						},
						Algorithm: algorithm,
					}},
				},
			}

			err := validateConfigStructure(cfg)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("apply to executors; " + algorithm.Type + " is a selector"))
		}
	})

	It("rejects an executor budget without a deadline", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "ratings-without-deadline",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type:    "ratings",
						Budget:  &AlgorithmBudget{MaxCalls: 2},
						Ratings: &RatingsAlgorithmConfig{},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("algorithm.budget.deadline"))
	})
}

func registerValidateConfigStructureWorkflowsSpecs() {
	registerValidateConfigStructureFlowDefaultsSpecs()
	registerValidateConfigStructureDynamicWorkflowsSpecs()
	registerValidateConfigStructureStaticWorkflowsSpecs()
}

func registerValidateConfigStructureFlowDefaultsSpecs() {
	It("does not reserve a Flow model name", func() { Expect((&RouterConfig{}).IsEntrypointModelName(DefaultFlowModelName)).To(BeFalse()) })
}

func registerValidateConfigStructureDynamicWorkflowsSpecs() {
	registerValidateConfigStructureDynamicWorkflowPlannerSpecs()
	registerValidateConfigStructureDynamicWorkflowFinalSpecs()
}

func registerValidateConfigStructureDynamicWorkflowPlannerSpecs() {
	It("accepts dynamic workflows with planner model", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "dynamic-flow",
					ModelRefs: []ModelRef{{
						Model:                 "worker-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type: "workflows",
						Workflows: &WorkflowsAlgorithmConfig{
							Mode: WorkflowModeDynamic,
							Planner: WorkflowPlannerConfig{
								Model:               "qwen-coordinator",
								MaxCompletionTokens: 1024,
							},
							MaxSteps:               4,
							MaxParallel:            2,
							RoundTimeoutSeconds:    90,
							MinSuccessfulResponses: 1,
						},
					},
				}},
			},
		}

		Expect(validateConfigStructure(withDecisionBackends(cfg))).To(Succeed())
	})

	It("rejects dynamic workflows with invalid planner max completion tokens", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "dynamic-flow",
					ModelRefs: []ModelRef{{
						Model:                 "worker-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type: "workflows",
						Workflows: &WorkflowsAlgorithmConfig{
							Mode: WorkflowModeDynamic,
							Planner: WorkflowPlannerConfig{
								Model:               "qwen-coordinator",
								MaxCompletionTokens: -1,
							},
						},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("planner.max_completion_tokens must be >= 1"))
	})

	It("rejects dynamic workflows with invalid quorum controls", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "dynamic-flow-invalid-quorum",
					ModelRefs: []ModelRef{{
						Model:                 "worker-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type: "workflows",
						Workflows: &WorkflowsAlgorithmConfig{
							Mode:                   WorkflowModeDynamic,
							Planner:                WorkflowPlannerConfig{Model: "qwen-coordinator"},
							RoundTimeoutSeconds:    -1,
							MinSuccessfulResponses: 1,
						},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("round_timeout_seconds must be >= 1"))
	})
}

func registerValidateConfigStructureDynamicWorkflowFinalSpecs() {
	It("accepts dynamic workflows with configured final model", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "dynamic-flow-final",
					ModelRefs: []ModelRef{
						{
							Model:                 "worker-a",
							ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
						},
						{
							Model:                 "final-a",
							ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
						},
					},
					Algorithm: &AlgorithmConfig{
						Type: "workflows",
						Workflows: &WorkflowsAlgorithmConfig{
							Mode:    WorkflowModeDynamic,
							Planner: WorkflowPlannerConfig{Model: "qwen-coordinator"},
							Final:   WorkflowFinalConfig{Model: "final-a"},
						},
					},
				}},
			},
		}

		Expect(validateConfigStructure(withDecisionBackends(cfg))).To(Succeed())
	})

	It("rejects dynamic workflow final model outside modelRefs", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "dynamic-flow-final-outside-modelrefs",
					ModelRefs: []ModelRef{{
						Model:                 "worker-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type: "workflows",
						Workflows: &WorkflowsAlgorithmConfig{
							Mode:    WorkflowModeDynamic,
							Planner: WorkflowPlannerConfig{Model: "qwen-coordinator"},
							Final:   WorkflowFinalConfig{Model: "final-a"},
						},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("algorithm.workflows.final.model references model \"final-a\" outside decision modelRefs"))
	})

	It("accepts dynamic workflows with an assigned worker planner default", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "dynamic-flow-no-planner",
					ModelRefs: []ModelRef{{
						Model:                 "worker-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type: "workflows",
						Workflows: &WorkflowsAlgorithmConfig{
							Mode: WorkflowModeDynamic,
						},
					},
				}},
			},
		}

		err := validateConfigStructure(withDecisionBackends(cfg))
		Expect(err).NotTo(HaveOccurred())
	})
}

func registerValidateConfigStructureStaticWorkflowsSpecs() {
	It("accepts static workflows with explicit roles", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "static-flow",
					ModelRefs: []ModelRef{
						{
							Model:                 "worker-a",
							ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
						},
						{
							Model:                 "worker-b",
							ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
						},
					},
					Algorithm: &AlgorithmConfig{
						Type: "workflows",
						Workflows: &WorkflowsAlgorithmConfig{
							Mode: WorkflowModeStatic,
							Roles: []WorkflowRoleConfig{
								{Name: "thinker", Models: []string{"worker-a"}},
								{Name: "verifier", Models: []string{"worker-b"}},
							},
							Final: WorkflowFinalConfig{Model: "worker-b"},
						},
					},
				}},
			},
		}

		Expect(validateConfigStructure(withDecisionBackends(cfg))).To(Succeed())
	})

	It("rejects static workflows without roles", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "static-flow-no-roles",
					ModelRefs: []ModelRef{{
						Model:                 "worker-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type:      "workflows",
						Workflows: &WorkflowsAlgorithmConfig{Mode: WorkflowModeStatic},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("algorithm.workflows: roles is required when mode=static"))
	})

	It("rejects static workflow role models outside modelRefs", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "static-flow-outside-modelrefs",
					ModelRefs: []ModelRef{{
						Model:                 "worker-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type: "workflows",
						Workflows: &WorkflowsAlgorithmConfig{
							Mode: WorkflowModeStatic,
							Roles: []WorkflowRoleConfig{{
								Name:   "worker",
								Models: []string{"worker-b"},
							}},
						},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("references model \"worker-b\" outside decision modelRefs"))
	})
}

func registerValidateConfigStructureAlgorithmTypeMismatchSpecs() {
	It("rejects algorithm type and config block mismatch", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "mismatched-algo-block",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type:    "fusion",
						Ratings: &RatingsAlgorithmConfig{},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("requires algorithm.fusion configuration; found algorithm.ratings"))
	})

	It("rejects unsupported algorithm block for static type", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "static-with-block",
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{
						Type:        "static",
						MultiFactor: &MultiFactorSelectionConfig{},
					},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("algorithm.type=static cannot be used with algorithm.multi_factor configuration"))
	})
}

func registerValidateConfigStructureSignalReferenceSpecs() {
	It("rejects unsupported signal condition types", func() {
		cfg := &RouterConfig{
			IntelligentRouting: IntelligentRouting{
				Decisions: []Decision{{
					Name: "unsupported-signal",
					Rules: RuleCombination{
						Operator: "AND",
						Conditions: []RuleCondition{
							{Type: "unknown", Name: "unknown-signal"},
						},
					},
					ModelRefs: []ModelRef{{
						Model:                 "model-a",
						ModelReasoningControl: ModelReasoningControl{UseReasoning: boolPtr(true)},
					}},
					Algorithm: &AlgorithmConfig{Type: "static"},
				}},
			},
		}

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("unsupported signal type \"unknown\""))
	})
}

var _ = Describe("validateConfigStructure", func() {
	registerValidateConfigStructureCoreSpecs()
	registerValidateConfigStructureLoRASpecs()
	registerValidateConfigStructureAlgorithmSpecs()
})
