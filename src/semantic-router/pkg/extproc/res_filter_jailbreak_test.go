package extproc

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

var _ = Describe("Response Jailbreak Filter", func() {
	var (
		router *OpenAIRouter
		cfg    *config.RouterConfig
	)

	createDecisionWithResponseJailbreak := func(enabled bool, action string) *config.Decision {
		return &config.Decision{
			Name: "test_decision",
			Plugins: []config.DecisionPlugin{
				{
					Type: "response_jailbreak",
					Configuration: config.MustStructuredPayload(map[string]interface{}{
						"enabled":   enabled,
						"threshold": 0.5,
						"action":    action,
					}),
				},
			},
		}
	}

	BeforeEach(func() {
		cfg = &config.RouterConfig{}
		router = &OpenAIRouter{
			Config: cfg,
		}
	})

	Describe("shouldPerformResponseJailbreakDetection", func() {
		It("should return false when classifier is nil", func() {
			ctx := &RequestContext{
				VSRSelectedDecision: createDecisionWithResponseJailbreak(true, "header"),
			}
			Expect(router.shouldPerformResponseJailbreakDetection(ctx)).To(BeFalse())
		})

		It("should return false when decision is nil", func() {
			ctx := &RequestContext{
				VSRSelectedDecision: nil,
			}
			Expect(router.shouldPerformResponseJailbreakDetection(ctx)).To(BeFalse())
		})

		It("should return false when plugin not enabled", func() {
			ctx := &RequestContext{
				VSRSelectedDecision: createDecisionWithResponseJailbreak(false, "header"),
			}
			Expect(router.shouldPerformResponseJailbreakDetection(ctx)).To(BeFalse())
		})

		It("should return false when no response_jailbreak plugin configured", func() {
			decision := &config.Decision{
				Name:    "test_decision",
				Plugins: []config.DecisionPlugin{},
			}
			ctx := &RequestContext{
				VSRSelectedDecision: decision,
			}
			Expect(router.shouldPerformResponseJailbreakDetection(ctx)).To(BeFalse())
		})
	})

	Describe("getResponseJailbreakAction", func() {
		It("should return 'header' when decision is nil", func() {
			Expect(router.getResponseJailbreakAction(nil)).To(Equal("header"))
		})

		It("should return 'header' when no plugin configured", func() {
			decision := &config.Decision{
				Name:    "test_decision",
				Plugins: []config.DecisionPlugin{},
			}
			Expect(router.getResponseJailbreakAction(decision)).To(Equal("header"))
		})

		It("should return 'header' when action not specified", func() {
			decision := &config.Decision{
				Name: "test_decision",
				Plugins: []config.DecisionPlugin{
					{
						Type: "response_jailbreak",
						Configuration: config.MustStructuredPayload(map[string]interface{}{
							"enabled": true,
						}),
					},
				},
			}
			Expect(router.getResponseJailbreakAction(decision)).To(Equal("header"))
		})

		It("should return 'block' when action is block", func() {
			Expect(router.getResponseJailbreakAction(
				createDecisionWithResponseJailbreak(true, "block"),
			)).To(Equal("block"))
		})

		It("should return 'none' when action is none", func() {
			Expect(router.getResponseJailbreakAction(
				createDecisionWithResponseJailbreak(true, "none"),
			)).To(Equal("none"))
		})

		It("should return 'header' when action is header", func() {
			Expect(router.getResponseJailbreakAction(
				createDecisionWithResponseJailbreak(true, "header"),
			)).To(Equal("header"))
		})
	})

	Describe("responseJailbreakWarningCode", func() {
		It("returns no code when no jailbreak detected", func() {
			ctx := &RequestContext{ResponseJailbreakDetected: false}
			Expect(router.responseJailbreakWarningCode(ctx)).To(BeEmpty())
		})

		It("returns the response_jailbreak code when action is header", func() {
			ctx := &RequestContext{
				ResponseJailbreakDetected:   true,
				ResponseJailbreakType:       "entity_redirection",
				ResponseJailbreakConfidence: 0.85,
				VSRSelectedDecision:         createDecisionWithResponseJailbreak(true, "header"),
			}
			Expect(router.responseJailbreakWarningCode(ctx)).To(Equal(headers.ResponseWarningJailbreak))
		})

		It("returns no code when action is none", func() {
			ctx := &RequestContext{
				ResponseJailbreakDetected:   true,
				ResponseJailbreakType:       "entity_redirection",
				ResponseJailbreakConfidence: 0.85,
				VSRSelectedDecision:         createDecisionWithResponseJailbreak(true, "none"),
			}
			Expect(router.responseJailbreakWarningCode(ctx)).To(BeEmpty())
		})
	})

	Describe("GetResponseJailbreakConfig", func() {
		It("should return nil when no plugin configured", func() {
			decision := &config.Decision{
				Name:    "test",
				Plugins: []config.DecisionPlugin{},
			}
			Expect(decision.GetResponseJailbreakConfig()).To(BeNil())
		})

		It("should parse config correctly", func() {
			decision := createDecisionWithResponseJailbreak(true, "block")
			rjCfg := decision.GetResponseJailbreakConfig()
			Expect(rjCfg).NotTo(BeNil())
			Expect(rjCfg.Enabled).To(BeTrue())
			Expect(rjCfg.Threshold).To(BeNumerically("~", 0.5, 0.01))
			Expect(rjCfg.Action).To(Equal("block"))
		})
	})
})
