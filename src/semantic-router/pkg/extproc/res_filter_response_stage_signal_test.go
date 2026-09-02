package extproc

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("Response stage jailbreak signal", func() {
	newRouter := func(rules ...config.JailbreakRule) *OpenAIRouter {
		cfg := &config.RouterConfig{}
		cfg.Signals.JailbreakRules = rules
		return &OpenAIRouter{Config: cfg}
	}
	responseRule := config.JailbreakRule{Name: "unsafe_completion", Threshold: 0.7, Direction: config.SignalDirectionResponse}

	// The signal is driven by the declared rules, not by an enforcement plugin.
	// A decision has to be able to read it whether or not response_jailbreak is
	// enabled anywhere, or it is plugin state wearing a signal's name.
	It("publishes without any plugin configured", func() {
		router := newRouter(responseRule)
		ctx := &RequestContext{} // no VSRSelectedDecision, so no plugin at all

		router.evaluateResponseJailbreakSignal(ctx, "some answer")

		// No classifier is wired here, so the rule resolves as unavailable
		// rather than silently absent, under the plain jailbreak key.
		Expect(ctx.VSRSignalErrors).To(HaveKey("jailbreak:unsafe_completion"))
		Expect(ctx.VSRMatchedResponseJailbreak).To(BeEmpty())
	})

	It("does nothing when only request-direction rules are declared", func() {
		router := newRouter(config.JailbreakRule{Name: "prompt_injection", Threshold: 0.7})
		ctx := &RequestContext{}

		router.evaluateResponseJailbreakSignal(ctx, "some answer")

		Expect(ctx.VSRSignalErrors).To(BeEmpty())
		Expect(ctx.VSRSignalConfidences).To(BeEmpty())
	})

	It("reports a response with no assistant text as unresolved, not clean", func() {
		router := newRouter(responseRule)
		ctx := &RequestContext{}

		router.evaluateResponseJailbreakSignal(ctx, "")

		Expect(ctx.VSRSignalErrors).To(HaveKey("jailbreak:unsafe_completion"))
	})

	It("reads the outcome from the response-direction rules only", func() {
		router := newRouter(responseRule, config.JailbreakRule{Name: "prompt_injection", Threshold: 0.7})
		ctx := &RequestContext{
			// A failed request-stage scan under the same type must not make
			// the response scan look unresolved.
			VSRSignalErrors:             map[string]string{"jailbreak:prompt_injection": "jailbreak_evaluation_failed"},
			VSRMatchedResponseJailbreak: []string{"unsafe_completion"},
		}

		matched, resolved := router.responseJailbreakSignalOutcome(ctx)
		Expect(resolved).To(BeTrue())
		Expect(matched).To(BeTrue())
	})

	Describe("lowestResponseJailbreakThreshold", func() {
		It("takes the most permissive rule so one call serves them all", func() {
			Expect(lowestResponseJailbreakThreshold([]config.JailbreakRule{
				{Name: "lenient", Threshold: 0.9},
				{Name: "strict", Threshold: 0.4},
			})).To(BeNumerically("~", 0.4, 1e-6))
		})
	})
})
