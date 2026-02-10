package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
)

var _ = Describe("ModalityDetectionConfig", func() {

	Describe("Validate", func() {

		// ── nil / empty ─────────────────────────────────────────────
		It("accepts nil config (defaults to hybrid at runtime)", func() {
			var cfg *ModalityDetectionConfig
			Expect(cfg.Validate()).To(Succeed())
		})

		// ── method: classifier ──────────────────────────────────────
		Context("method=classifier", func() {
			It("accepts valid classifier config", func() {
				cfg := &ModalityDetectionConfig{
					Method: ModalityDetectionClassifier,
					Classifier: &ModalityClassifierConfig{
						ModelPath: "./models/mmbert32k-modality-router-merged",
					},
					ConfidenceThreshold: 0.6,
				}
				Expect(cfg.Validate()).To(Succeed())
			})

			It("rejects missing classifier block", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionClassifier,
					ConfidenceThreshold: 0.6,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("classifier.model_path"))
			})

			It("rejects empty model_path", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionClassifier,
					Classifier:          &ModalityClassifierConfig{ModelPath: ""},
					ConfidenceThreshold: 0.6,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("classifier.model_path"))
			})

			It("does not require keywords when method is classifier", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionClassifier,
					Classifier:          &ModalityClassifierConfig{ModelPath: "/some/model"},
					ConfidenceThreshold: 0.7,
					// Keywords intentionally omitted
				}
				Expect(cfg.Validate()).To(Succeed())
			})

			It("rejects missing confidence_threshold", func() {
				cfg := &ModalityDetectionConfig{
					Method:     ModalityDetectionClassifier,
					Classifier: &ModalityClassifierConfig{ModelPath: "/some/model"},
					// ConfidenceThreshold intentionally omitted (zero value)
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("confidence_threshold is required"))
			})
		})

		// ── method: keyword ─────────────────────────────────────────
		Context("method=keyword", func() {
			It("accepts valid keyword config", func() {
				cfg := &ModalityDetectionConfig{
					Method:   ModalityDetectionKeyword,
					Keywords: []string{"generate an image", "draw"},
				}
				Expect(cfg.Validate()).To(Succeed())
			})

			It("accepts keyword config with both_keywords", func() {
				cfg := &ModalityDetectionConfig{
					Method:       ModalityDetectionKeyword,
					Keywords:     []string{"generate an image"},
					BothKeywords: []string{"and illustrate"},
				}
				Expect(cfg.Validate()).To(Succeed())
			})

			It("rejects empty keywords list", func() {
				cfg := &ModalityDetectionConfig{
					Method:   ModalityDetectionKeyword,
					Keywords: []string{},
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("requires at least one entry in keywords"))
			})

			It("rejects nil keywords", func() {
				cfg := &ModalityDetectionConfig{
					Method: ModalityDetectionKeyword,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("keywords"))
			})

			It("does not require classifier when method is keyword", func() {
				cfg := &ModalityDetectionConfig{
					Method:   ModalityDetectionKeyword,
					Keywords: []string{"draw"},
					// Classifier intentionally omitted
				}
				Expect(cfg.Validate()).To(Succeed())
			})
		})

		// ── method: hybrid ──────────────────────────────────────────
		Context("method=hybrid", func() {
			It("accepts hybrid with both classifier and keywords", func() {
				cfg := &ModalityDetectionConfig{
					Method: ModalityDetectionHybrid,
					Classifier: &ModalityClassifierConfig{
						ModelPath: "./models/mmbert32k",
					},
					Keywords:            []string{"generate an image"},
					ConfidenceThreshold: 0.6,
					LowerThresholdRatio: 0.7,
				}
				Expect(cfg.Validate()).To(Succeed())
			})

			It("accepts hybrid with classifier only", func() {
				cfg := &ModalityDetectionConfig{
					Method: ModalityDetectionHybrid,
					Classifier: &ModalityClassifierConfig{
						ModelPath: "./models/mmbert32k",
					},
					ConfidenceThreshold: 0.6,
					LowerThresholdRatio: 0.7,
				}
				Expect(cfg.Validate()).To(Succeed())
			})

			It("accepts hybrid with keywords only", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionHybrid,
					Keywords:            []string{"draw a picture"},
					ConfidenceThreshold: 0.5,
					LowerThresholdRatio: 0.7,
				}
				Expect(cfg.Validate()).To(Succeed())
			})

			It("rejects hybrid with neither classifier nor keywords", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionHybrid,
					ConfidenceThreshold: 0.6,
					LowerThresholdRatio: 0.7,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("at least one of classifier.model_path or keywords"))
			})

			It("rejects hybrid when classifier has empty model_path and no keywords", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionHybrid,
					Classifier:          &ModalityClassifierConfig{ModelPath: ""},
					ConfidenceThreshold: 0.6,
					LowerThresholdRatio: 0.7,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("at least one of"))
			})

			It("rejects empty method", func() {
				cfg := &ModalityDetectionConfig{
					// Method intentionally empty — should be rejected, not default
					Keywords:            []string{"generate"},
					ConfidenceThreshold: 0.6,
					LowerThresholdRatio: 0.7,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("method is required"))
			})

			It("rejects empty method even when thresholds are set", func() {
				cfg := &ModalityDetectionConfig{
					ConfidenceThreshold: 0.6,
					LowerThresholdRatio: 0.7,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("method is required"))
			})

			It("rejects hybrid without confidence_threshold", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionHybrid,
					Keywords:            []string{"draw"},
					LowerThresholdRatio: 0.7,
					// ConfidenceThreshold intentionally omitted
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("confidence_threshold is required"))
			})

			It("rejects hybrid without lower_threshold_ratio", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionHybrid,
					Keywords:            []string{"draw"},
					ConfidenceThreshold: 0.6,
					// LowerThresholdRatio intentionally omitted
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("lower_threshold_ratio is required"))
			})

			It("rejects lower_threshold_ratio > 1", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionHybrid,
					Keywords:            []string{"draw"},
					ConfidenceThreshold: 0.6,
					LowerThresholdRatio: 1.5,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("lower_threshold_ratio"))
			})

			It("rejects negative lower_threshold_ratio", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionHybrid,
					Keywords:            []string{"draw"},
					ConfidenceThreshold: 0.6,
					LowerThresholdRatio: -0.2,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("lower_threshold_ratio"))
			})

			It("does not require lower_threshold_ratio for classifier method", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionClassifier,
					Classifier:          &ModalityClassifierConfig{ModelPath: "/model"},
					ConfidenceThreshold: 0.6,
					// LowerThresholdRatio not needed for classifier-only
				}
				Expect(cfg.Validate()).To(Succeed())
			})
		})

		// ── invalid method ──────────────────────────────────────────
		Context("invalid method", func() {
			It("rejects unknown method", func() {
				cfg := &ModalityDetectionConfig{
					Method: "regex",
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("must be one of"))
				Expect(err.Error()).To(ContainSubstring("regex"))
			})

			It("rejects method with wrong capitalization", func() {
				cfg := &ModalityDetectionConfig{
					Method: "Classifier",
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("must be one of"))
			})
		})

		// ── confidence_threshold ─────────────────────────────────────
		Context("confidence_threshold", func() {
			It("accepts valid threshold for keyword method", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionKeyword,
					Keywords:            []string{"draw"},
					ConfidenceThreshold: 0.7,
				}
				Expect(cfg.Validate()).To(Succeed())
			})

			It("accepts threshold of exactly 1.0", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionKeyword,
					Keywords:            []string{"draw"},
					ConfidenceThreshold: 1.0,
				}
				Expect(cfg.Validate()).To(Succeed())
			})

			It("accepts zero threshold for keyword method (not required)", func() {
				cfg := &ModalityDetectionConfig{
					Method:   ModalityDetectionKeyword,
					Keywords: []string{"draw"},
					// ConfidenceThreshold omitted — not required for keyword
				}
				Expect(cfg.Validate()).To(Succeed())
			})

			It("rejects zero threshold for classifier method (required)", func() {
				cfg := &ModalityDetectionConfig{
					Method:     ModalityDetectionClassifier,
					Classifier: &ModalityClassifierConfig{ModelPath: "/model"},
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("confidence_threshold is required"))
			})

			It("rejects zero threshold for hybrid method (required)", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionHybrid,
					Keywords:            []string{"draw"},
					LowerThresholdRatio: 0.7,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("confidence_threshold is required"))
			})

			It("rejects negative threshold", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionKeyword,
					Keywords:            []string{"draw"},
					ConfidenceThreshold: -0.1,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("confidence_threshold"))
			})

			It("rejects threshold greater than 1", func() {
				cfg := &ModalityDetectionConfig{
					Method:              ModalityDetectionKeyword,
					Keywords:            []string{"draw"},
					ConfidenceThreshold: 1.5,
				}
				err := cfg.Validate()
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("confidence_threshold"))
			})
		})
	})

	Describe("GetMethod", func() {
		It("returns empty for nil config", func() {
			var cfg *ModalityDetectionConfig
			Expect(cfg.GetMethod()).To(Equal(""))
		})

		It("returns empty for empty method", func() {
			cfg := &ModalityDetectionConfig{}
			Expect(cfg.GetMethod()).To(Equal(""))
		})

		It("returns the configured method", func() {
			cfg := &ModalityDetectionConfig{Method: ModalityDetectionClassifier}
			Expect(cfg.GetMethod()).To(Equal(ModalityDetectionClassifier))
		})
	})

	Describe("GetConfidenceThreshold", func() {
		It("returns 0 for nil config", func() {
			var cfg *ModalityDetectionConfig
			Expect(cfg.GetConfidenceThreshold()).To(BeNumerically("==", 0))
		})

		It("returns 0 for zero threshold (unset)", func() {
			cfg := &ModalityDetectionConfig{}
			Expect(cfg.GetConfidenceThreshold()).To(BeNumerically("==", 0))
		})

		It("returns configured threshold", func() {
			cfg := &ModalityDetectionConfig{ConfidenceThreshold: 0.85}
			Expect(cfg.GetConfidenceThreshold()).To(BeNumerically("~", 0.85, 0.001))
		})
	})

	Describe("GetLowerThresholdRatio", func() {
		It("returns 0 for nil config", func() {
			var cfg *ModalityDetectionConfig
			Expect(cfg.GetLowerThresholdRatio()).To(BeNumerically("==", 0))
		})

		It("returns 0 for zero ratio (unset)", func() {
			cfg := &ModalityDetectionConfig{}
			Expect(cfg.GetLowerThresholdRatio()).To(BeNumerically("==", 0))
		})

		It("returns configured ratio", func() {
			cfg := &ModalityDetectionConfig{LowerThresholdRatio: 0.7}
			Expect(cfg.GetLowerThresholdRatio()).To(BeNumerically("~", 0.7, 0.001))
		})
	})
})

var _ = Describe("ImageGenPluginConfig.Validate", func() {

	It("passes when disabled", func() {
		cfg := &ImageGenPluginConfig{Enabled: false}
		Expect(cfg.Validate()).To(Succeed())
	})

	It("rejects enabled config with empty backend", func() {
		cfg := &ImageGenPluginConfig{Enabled: true}
		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("backend is required"))
	})

	It("rejects unknown backend", func() {
		cfg := &ImageGenPluginConfig{Enabled: true, Backend: "stable_diffusion"}
		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("unknown image_gen backend"))
	})

	Context("with vllm_omni backend", func() {
		It("passes with valid vllm_omni + valid modality detection", func() {
			cfg := &ImageGenPluginConfig{
				Enabled:       true,
				Backend:       "vllm_omni",
				BackendConfig: &VLLMOmniImageGenConfig{BaseURL: "http://localhost:8001"},
				ModalityDetection: &ModalityDetectionConfig{
					Method:   ModalityDetectionKeyword,
					Keywords: []string{"generate an image"},
				},
			}
			Expect(cfg.Validate()).To(Succeed())
		})

		It("rejects when modality detection is invalid", func() {
			cfg := &ImageGenPluginConfig{
				Enabled:       true,
				Backend:       "vllm_omni",
				BackendConfig: &VLLMOmniImageGenConfig{BaseURL: "http://localhost:8001"},
				ModalityDetection: &ModalityDetectionConfig{
					Method: ModalityDetectionClassifier,
					// Missing classifier config
				},
			}
			err := cfg.Validate()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("image_gen"))
			Expect(err.Error()).To(ContainSubstring("classifier.model_path"))
		})
	})

	It("passes with nil modality detection (defaults apply at runtime)", func() {
		cfg := &ImageGenPluginConfig{
			Enabled:       true,
			Backend:       "vllm_omni",
			BackendConfig: &VLLMOmniImageGenConfig{BaseURL: "http://localhost:8001"},
			// ModalityDetection intentionally nil
		}
		Expect(cfg.Validate()).To(Succeed())
	})
})

var _ = Describe("Config load validates modality_detection in image_gen plugin", func() {

	// Helper to create a minimal valid config YAML with an image_gen plugin
	buildConfigYAML := func(modalityBlock string) string {
		return `
endpoints:
  - name: vllm-test
    url: http://localhost:8001
    models:
      - name: test-model
model_config:
  test-model:
    endpoints:
      - vllm-test
decisions:
  - name: test_decision
    modelRefs:
      - model: test-model
        use_reasoning: false
    plugins:
      - type: image_gen
        configuration:
          enabled: true
          backend: vllm_omni
          backend_config:
            base_url: http://localhost:8001
` + modalityBlock
	}

	It("rejects classifier method without model_path during config load", func() {
		cfgYAML := buildConfigYAML(`          modality_detection:
            method: classifier`)
		cfg := &RouterConfig{}
		err := yaml.Unmarshal([]byte(cfgYAML), cfg)
		Expect(err).NotTo(HaveOccurred())

		err = validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("classifier.model_path"))
	})

	It("rejects keyword method without keywords during config load", func() {
		cfgYAML := buildConfigYAML(`          modality_detection:
            method: keyword`)
		cfg := &RouterConfig{}
		err := yaml.Unmarshal([]byte(cfgYAML), cfg)
		Expect(err).NotTo(HaveOccurred())

		err = validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("keywords"))
	})

	It("rejects invalid method during config load", func() {
		cfgYAML := buildConfigYAML(`          modality_detection:
            method: regex`)
		cfg := &RouterConfig{}
		err := yaml.Unmarshal([]byte(cfgYAML), cfg)
		Expect(err).NotTo(HaveOccurred())

		err = validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("must be one of"))
	})

	It("accepts valid keyword config during config load", func() {
		cfgYAML := buildConfigYAML(`          modality_detection:
            method: keyword
            keywords:
              - "generate an image"
              - "draw a picture"`)
		cfg := &RouterConfig{}
		err := yaml.Unmarshal([]byte(cfgYAML), cfg)
		Expect(err).NotTo(HaveOccurred())

		err = validateConfigStructure(cfg)
		Expect(err).NotTo(HaveOccurred())
	})

	It("accepts valid classifier config during config load", func() {
		cfgYAML := buildConfigYAML(`          modality_detection:
            method: classifier
            confidence_threshold: 0.7
            classifier:
              model_path: ./models/mmbert32k-modality-router-merged`)
		cfg := &RouterConfig{}
		err := yaml.Unmarshal([]byte(cfgYAML), cfg)
		Expect(err).NotTo(HaveOccurred())

		err = validateConfigStructure(cfg)
		Expect(err).NotTo(HaveOccurred())
	})

	It("accepts valid hybrid config during config load", func() {
		cfgYAML := buildConfigYAML(`          modality_detection:
            method: hybrid
            classifier:
              model_path: ./models/mmbert32k-modality-router-merged
            keywords:
              - "generate an image"
            both_keywords:
              - "and illustrate"
            confidence_threshold: 0.7
            lower_threshold_ratio: 0.7`)
		cfg := &RouterConfig{}
		err := yaml.Unmarshal([]byte(cfgYAML), cfg)
		Expect(err).NotTo(HaveOccurred())

		err = validateConfigStructure(cfg)
		Expect(err).NotTo(HaveOccurred())
	})

	It("rejects hybrid missing lower_threshold_ratio during config load", func() {
		cfgYAML := buildConfigYAML(`          modality_detection:
            method: hybrid
            keywords:
              - "generate an image"
            confidence_threshold: 0.7`)
		cfg := &RouterConfig{}
		err := yaml.Unmarshal([]byte(cfgYAML), cfg)
		Expect(err).NotTo(HaveOccurred())

		err = validateConfigStructure(cfg)
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("lower_threshold_ratio is required"))
	})
})
