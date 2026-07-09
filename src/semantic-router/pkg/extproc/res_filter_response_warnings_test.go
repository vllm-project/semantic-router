package extproc

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// decisionWithHallucinationActions builds a decision whose hallucination plugin
// pins both the hallucination and unverified-factual actions, so the per-action
// branches of the warning appliers are exercised deterministically.
func decisionWithHallucinationActions(hallucinationAction, unverifiedAction string) *config.Decision {
	return &config.Decision{
		Name: "test_decision",
		Plugins: []config.DecisionPlugin{
			{
				Type: "hallucination",
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled":                   true,
					"hallucination_action":      hallucinationAction,
					"unverified_factual_action": unverifiedAction,
				}),
			},
		},
	}
}

var _ = Describe("Response warning appliers", func() {
	var router *OpenAIRouter
	bodyWithContent := []byte(`{"choices":[{"message":{"content":"hello"}}]}`)

	BeforeEach(func() {
		router = &OpenAIRouter{Config: &config.RouterConfig{}}
	})

	Describe("applyHallucinationWarning", func() {
		It("returns no code when no hallucination detected", func() {
			ctx := &RequestContext{HallucinationDetected: false}
			resultBody, code := router.applyHallucinationWarning(ctx, bodyWithContent)
			Expect(code).To(BeEmpty())
			Expect(resultBody).To(Equal(bodyWithContent))
		})

		It("surfaces the hallucination code on the default (header) action", func() {
			ctx := &RequestContext{HallucinationDetected: true}
			resultBody, code := router.applyHallucinationWarning(ctx, bodyWithContent)
			Expect(code).To(Equal(headers.ResponseWarningHallucination))
			Expect(resultBody).To(Equal(bodyWithContent))
		})

		It("rewrites the body and emits no code on the body action", func() {
			ctx := &RequestContext{
				HallucinationDetected: true,
				VSRSelectedDecision:   decisionWithHallucinationActions("body", "header"),
			}
			resultBody, code := router.applyHallucinationWarning(ctx, bodyWithContent)
			Expect(code).To(BeEmpty())
			Expect(resultBody).NotTo(Equal(bodyWithContent))
		})

		It("emits no code and leaves the body on the none action", func() {
			ctx := &RequestContext{
				HallucinationDetected: true,
				VSRSelectedDecision:   decisionWithHallucinationActions("none", "header"),
			}
			resultBody, code := router.applyHallucinationWarning(ctx, bodyWithContent)
			Expect(code).To(BeEmpty())
			Expect(resultBody).To(Equal(bodyWithContent))
		})
	})

	Describe("applyUnverifiedFactualWarning", func() {
		It("returns no code when the response is not unverified", func() {
			ctx := &RequestContext{UnverifiedFactualResponse: false}
			resultBody, code := router.applyUnverifiedFactualWarning(ctx, bodyWithContent)
			Expect(code).To(BeEmpty())
			Expect(resultBody).To(Equal(bodyWithContent))
		})

		It("surfaces the unverified_factual code on the default (header) action", func() {
			ctx := &RequestContext{
				UnverifiedFactualResponse: true,
				FactCheckNeeded:           true,
			}
			resultBody, code := router.applyUnverifiedFactualWarning(ctx, bodyWithContent)
			Expect(code).To(Equal(headers.ResponseWarningUnverifiedFactual))
			// The header action leaves the body unchanged.
			Expect(resultBody).To(Equal(bodyWithContent))
		})

		It("rewrites the body and emits no code on the body action", func() {
			ctx := &RequestContext{
				UnverifiedFactualResponse: true,
				VSRSelectedDecision:       decisionWithHallucinationActions("header", "body"),
			}
			resultBody, code := router.applyUnverifiedFactualWarning(ctx, bodyWithContent)
			Expect(code).To(BeEmpty())
			Expect(resultBody).NotTo(Equal(bodyWithContent))
		})

		It("emits no code and leaves the body on the none action", func() {
			ctx := &RequestContext{
				UnverifiedFactualResponse: true,
				VSRSelectedDecision:       decisionWithHallucinationActions("header", "none"),
			}
			resultBody, code := router.applyUnverifiedFactualWarning(ctx, bodyWithContent)
			Expect(code).To(BeEmpty())
			Expect(resultBody).To(Equal(bodyWithContent))
		})
	})
})
