package extproc

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	responseWithContent := func() *llmprotocol.Response {
		return &llmprotocol.Response{Output: []llmprotocol.OutputItem{{
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentText,
				Text: "hello",
			}},
		}}}
	}

	BeforeEach(func() {
		router = &OpenAIRouter{Config: &config.RouterConfig{}}
	})

	Describe("applySemanticHallucinationWarning", func() {
		It("returns no code when no hallucination detected", func() {
			ctx := &RequestContext{HallucinationDetected: false}
			response := responseWithContent()
			changed, code := router.applySemanticHallucinationWarning(ctx, response)
			Expect(code).To(BeEmpty())
			Expect(changed).To(BeFalse())
			Expect(semanticAssistantContent(response)).To(Equal("hello"))
		})

		It("surfaces the hallucination code on the default (header) action", func() {
			ctx := &RequestContext{HallucinationDetected: true}
			response := responseWithContent()
			changed, code := router.applySemanticHallucinationWarning(ctx, response)
			Expect(code).To(Equal(headers.ResponseWarningHallucination))
			Expect(changed).To(BeFalse())
			Expect(semanticAssistantContent(response)).To(Equal("hello"))
		})

		It("rewrites the body and emits no code on the body action", func() {
			ctx := &RequestContext{
				HallucinationDetected: true,
				VSRSelectedDecision:   decisionWithHallucinationActions("body", "header"),
			}
			response := responseWithContent()
			changed, code := router.applySemanticHallucinationWarning(ctx, response)
			Expect(code).To(BeEmpty())
			Expect(changed).To(BeTrue())
			Expect(semanticAssistantContent(response)).To(ContainSubstring("Hallucination Warning"))
		})

		It("emits no code and leaves the body on the none action", func() {
			ctx := &RequestContext{
				HallucinationDetected: true,
				VSRSelectedDecision:   decisionWithHallucinationActions("none", "header"),
			}
			response := responseWithContent()
			changed, code := router.applySemanticHallucinationWarning(ctx, response)
			Expect(code).To(BeEmpty())
			Expect(changed).To(BeFalse())
			Expect(semanticAssistantContent(response)).To(Equal("hello"))
		})
	})

	Describe("applySemanticUnverifiedFactualWarning", func() {
		It("returns no code when the response is not unverified", func() {
			ctx := &RequestContext{UnverifiedFactualResponse: false}
			response := responseWithContent()
			changed, code := router.applySemanticUnverifiedFactualWarning(ctx, response)
			Expect(code).To(BeEmpty())
			Expect(changed).To(BeFalse())
			Expect(semanticAssistantContent(response)).To(Equal("hello"))
		})

		It("surfaces the unverified_factual code on the default (header) action", func() {
			ctx := &RequestContext{
				UnverifiedFactualResponse: true,
				FactCheckNeeded:           true,
			}
			response := responseWithContent()
			changed, code := router.applySemanticUnverifiedFactualWarning(ctx, response)
			Expect(code).To(Equal(headers.ResponseWarningUnverifiedFactual))
			Expect(changed).To(BeFalse())
			Expect(semanticAssistantContent(response)).To(Equal("hello"))
		})

		It("rewrites the body and emits no code on the body action", func() {
			ctx := &RequestContext{
				UnverifiedFactualResponse: true,
				VSRSelectedDecision:       decisionWithHallucinationActions("header", "body"),
			}
			response := responseWithContent()
			changed, code := router.applySemanticUnverifiedFactualWarning(ctx, response)
			Expect(code).To(BeEmpty())
			Expect(changed).To(BeTrue())
			Expect(semanticAssistantContent(response)).To(ContainSubstring("Unverified Response"))
		})

		It("emits no code and leaves the body on the none action", func() {
			ctx := &RequestContext{
				UnverifiedFactualResponse: true,
				VSRSelectedDecision:       decisionWithHallucinationActions("header", "none"),
			}
			response := responseWithContent()
			changed, code := router.applySemanticUnverifiedFactualWarning(ctx, response)
			Expect(code).To(BeEmpty())
			Expect(changed).To(BeFalse())
			Expect(semanticAssistantContent(response)).To(Equal("hello"))
		})
	})
})
