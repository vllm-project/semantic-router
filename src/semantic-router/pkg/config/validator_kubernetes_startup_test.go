package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// These specs cover issue #3758: validateConfigStructure used to skip every
// global validator for ConfigSource: ConfigSourceKubernetes, so a bad static
// setting (nothing to do with the CRD-supplied routing state) was accepted at
// startup and only surfaced later, if at all. It now runs the subset of
// global validators that do not depend on Providers.Models, Decisions, or
// Signals, since those are the only fields CRD conversion populates
// (pkg/k8s/converter.go); everything else in a Kubernetes document is
// present from the static parse onward.
var _ = Describe("Kubernetes startup validation (issue #3758)", func() {
	It("rejects a bad static global setting before CRD conversion", func() {
		cfg := &RouterConfig{
			ConfigSource: ConfigSourceKubernetes,
			BackendModels: BackendModels{
				ModelConfig: map[string]ModelParams{},
			},
		}
		cfg.Memory.DefaultSimilarityThreshold = 1.5 // out of the [0.0, 1.0] range

		err := validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("global memory default_similarity_threshold"))
	})

	It("still tolerates absent routing state for the validators that need it", func() {
		cfg := &RouterConfig{
			ConfigSource: ConfigSourceKubernetes,
			IntelligentRouting: IntelligentRouting{
				// A CRD merge has not happened: no decisions, no signals.
				Decisions: nil,
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})

	It("resolves a remote classifier backend against ExternalModels alone, before CRD conversion", func() {
		// ExternalModels is a static document field; Providers.Models,
		// Decisions, and Signals are the only fields CRD conversion
		// populates, and all three stay at their zero value here.
		cfg := &RouterConfig{ConfigSource: ConfigSourceKubernetes}
		cfg.PromptGuard = PromptGuardConfig{
			Backend:              &RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPClassify, Model: "guard-model"},
			Enabled:              true,
			JailbreakMappingPath: "x",
		}
		cfg.CategoryModel = CategoryModel{
			Backend: &RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPClassify, Model: "classify-model"},
		}
		cfg.PIIModel = PIIModel{
			Backend: &RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPClassify, Model: "classify-model"},
		}

		// Without the ExternalModels entries, each backend's model name is
		// unresolved and validateConfigStructure must reject the document.
		Expect(validateConfigStructure(cfg)).NotTo(Succeed())

		cfg.ExternalModels = []ExternalModelConfig{
			{
				Name:          "guard-model",
				ModelName:     "guard-model",
				ModelRole:     ModelRoleGuardrail,
				ModelEndpoint: ClassifierVLLMEndpoint{Address: "guard.internal", Port: 9000},
			},
			{
				Name:          "classify-model",
				ModelName:     "classify-model",
				ModelRole:     ModelRoleClassification,
				ModelEndpoint: ClassifierVLLMEndpoint{Address: "classify.internal", Port: 9000},
			},
		}
		Expect(validateConfigStructure(cfg)).To(Succeed())
	})

	It("validates the same static setting after CRD conversion too", func() {
		cfg := &RouterConfig{ConfigSource: ConfigSourceKubernetes}
		cfg.Memory.DefaultSimilarityThreshold = 1.5

		err := ValidateKubernetesConfigContracts(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("global memory default_similarity_threshold"))
	})

	It("derives the full validator list from the registry with nothing dropped", func() {
		Expect(globalConfigValidators(true)).To(HaveLen(len(globalConfigValidatorRegistry)))
		Expect(globalConfigContractValidators).To(HaveLen(len(globalConfigValidatorRegistry)))
		Expect(len(globalConfigValidators(false))).To(BeNumerically("<", len(globalConfigValidatorRegistry)))
	})
})
