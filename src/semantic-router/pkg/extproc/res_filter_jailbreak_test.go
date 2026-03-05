package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
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
					Configuration: map[string]interface{}{
						"enabled":   enabled,
						"threshold": 0.5,
						"action":    action,
					},
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
						Configuration: map[string]interface{}{
							"enabled": true,
						},
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

	Describe("applyResponseJailbreakWarning", func() {
		It("should not modify response when no jailbreak detected", func() {
			ctx := &RequestContext{
				ResponseJailbreakDetected: false,
			}
			response := createMockBodyResponse()
			body := []byte(`{"choices":[{"message":{"content":"hello"}}]}`)

			resultBody, resultResp := router.applyResponseJailbreakWarning(response, ctx, body)
			Expect(resultBody).To(Equal(body))
			Expect(resultResp).To(Equal(response))
		})

		It("should add headers when action is header", func() {
			ctx := &RequestContext{
				ResponseJailbreakDetected:   true,
				ResponseJailbreakType:       "entity_redirection",
				ResponseJailbreakConfidence: 0.85,
				VSRSelectedDecision:         createDecisionWithResponseJailbreak(true, "header"),
			}
			response := createMockBodyResponse()
			body := []byte(`{"choices":[{"message":{"content":"hello"}}]}`)

			_, resultResp := router.applyResponseJailbreakWarning(response, ctx, body)

			bodyResp, ok := resultResp.Response.(*ext_proc.ProcessingResponse_ResponseBody)
			Expect(ok).To(BeTrue())
			Expect(bodyResp.ResponseBody.Response).NotTo(BeNil())
			Expect(bodyResp.ResponseBody.Response.HeaderMutation).NotTo(BeNil())

			foundDetected := false
			foundType := false
			foundConfidence := false
			for _, h := range bodyResp.ResponseBody.Response.HeaderMutation.SetHeaders {
				switch h.Header.Key {
				case headers.ResponseJailbreakDetected:
					foundDetected = true
					Expect(string(h.Header.RawValue)).To(Equal("true"))
				case headers.ResponseJailbreakType:
					foundType = true
					Expect(string(h.Header.RawValue)).To(Equal("entity_redirection"))
				case headers.ResponseJailbreakConfidence:
					foundConfidence = true
					Expect(string(h.Header.RawValue)).To(Equal("0.850"))
				}
			}
			Expect(foundDetected).To(BeTrue())
			Expect(foundType).To(BeTrue())
			Expect(foundConfidence).To(BeTrue())
		})

		It("should not add headers when action is none", func() {
			ctx := &RequestContext{
				ResponseJailbreakDetected:   true,
				ResponseJailbreakType:       "entity_redirection",
				ResponseJailbreakConfidence: 0.85,
				VSRSelectedDecision:         createDecisionWithResponseJailbreak(true, "none"),
			}
			response := createMockBodyResponse()
			body := []byte(`{"choices":[{"message":{"content":"hello"}}]}`)

			resultBody, _ := router.applyResponseJailbreakWarning(response, ctx, body)
			Expect(resultBody).To(Equal(body))
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
