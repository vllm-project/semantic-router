package extproc

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
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

	// The request path fails closed on a classify error when prompt_guard sets
	// on_error: block. The response path reuses the same prompt_guard backend,
	// so it must apply the same policy instead of serving an unverified
	// response - see #2918.
	Describe("responseJailbreakOnClassifyError", func() {
		It("tolerates the failure under on_error: allow", func() {
			ctx := &RequestContext{
				VSRSelectedDecision: createDecisionWithResponseJailbreak(true, "block"),
			}
			Expect(router.responseJailbreakOnClassifyError(ctx, false, "d", 0)).To(BeNil())
			Expect(ctx.ResponseJailbreakDetected).To(BeFalse())
			Expect(ctx.ResponseJailbreakType).To(BeEmpty())
		})

		It("blocks under on_error: block when the action is block", func() {
			ctx := &RequestContext{
				VSRSelectedDecision: createDecisionWithResponseJailbreak(true, "block"),
			}
			resp := router.responseJailbreakOnClassifyError(ctx, true, "d", 0)
			Expect(resp).NotTo(BeNil())
			Expect(ctx.ResponseJailbreakDetected).To(BeTrue())
			Expect(ctx.ResponseJailbreakType).To(Equal(classification.JailbreakClassificationErrorType))
			Expect(ctx.ResponseJailbreakConfidence).To(BeNumerically("~", 1.0, 0.001))
		})

		// A failure is reported through whatever the decision's action asks
		// for, exactly as a real detection is - it does not force a 403.
		It("warns instead of blocking when the action is header", func() {
			ctx := &RequestContext{
				VSRSelectedDecision: createDecisionWithResponseJailbreak(true, "header"),
			}
			Expect(router.responseJailbreakOnClassifyError(ctx, true, "d", 0)).To(BeNil())
			Expect(ctx.ResponseJailbreakDetected).To(BeTrue())
			Expect(router.responseJailbreakWarningCode(ctx)).To(Equal(headers.ResponseWarningJailbreak))
		})

		It("stays silent when the action is none", func() {
			ctx := &RequestContext{
				VSRSelectedDecision: createDecisionWithResponseJailbreak(true, "none"),
			}
			Expect(router.responseJailbreakOnClassifyError(ctx, true, "d", 0)).To(BeNil())
			Expect(ctx.ResponseJailbreakDetected).To(BeTrue())
			Expect(router.responseJailbreakWarningCode(ctx)).To(BeEmpty())
		})
	})

	Describe("responseJailbreakFailsClosed", func() {
		It("is false when no classifier config is available", func() {
			Expect(responseJailbreakFailsClosed(nil)).To(BeFalse())
		})

		It("is false under the default on_error", func() {
			Expect(responseJailbreakFailsClosed(&config.RouterConfig{})).To(BeFalse())
		})

		It("is true when prompt_guard sets on_error: block", func() {
			blocking := &config.RouterConfig{}
			blocking.PromptGuard.OnError = config.OnErrorBlock
			Expect(responseJailbreakFailsClosed(blocking)).To(BeTrue())
		})
	})
})
