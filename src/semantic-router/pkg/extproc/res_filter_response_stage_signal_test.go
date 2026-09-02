package extproc

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("Response stage jailbreak signal", func() {
	newRouter := func(rules ...config.ResponseJailbreakRule) *OpenAIRouter {
		cfg := &config.RouterConfig{}
		cfg.Signals.ResponseJailbreakRules = rules
		return &OpenAIRouter{Config: cfg}
	}

	// The signal is driven by the declared rules, not by an enforcement plugin.
	// A decision has to be able to read it whether or not response_jailbreak is
	// enabled anywhere, or it is plugin state wearing a signal's name.
	It("publishes without any plugin configured", func() {
		router := newRouter(config.ResponseJailbreakRule{Name: "unsafe_completion", Threshold: 0.7})
		ctx := &RequestContext{} // no VSRSelectedDecision, so no plugin at all

		router.evaluateResponseJailbreakSignal(ctx, "some answer")

		// No classifier is wired here, so the rule resolves as unavailable
		// rather than silently absent.
		Expect(ctx.VSRSignalErrors).To(HaveKey("response_jailbreak:unsafe_completion"))
		Expect(ctx.VSRMatchedResponseJailbreak).To(BeEmpty())
	})

	It("does nothing when no rules are declared", func() {
		router := newRouter()
		ctx := &RequestContext{}

		router.evaluateResponseJailbreakSignal(ctx, "some answer")

		Expect(ctx.VSRSignalErrors).To(BeEmpty())
		Expect(ctx.VSRSignalConfidences).To(BeEmpty())
	})

	It("reports a response with no assistant text as unresolved, not clean", func() {
		router := newRouter(config.ResponseJailbreakRule{Name: "unsafe_completion", Threshold: 0.7})
		ctx := &RequestContext{}

		router.evaluateResponseJailbreakSignal(ctx, "")

		Expect(ctx.VSRSignalErrors).To(HaveKey("response_jailbreak:unsafe_completion"))
	})

	Describe("lowestResponseJailbreakThreshold", func() {
		It("takes the most permissive rule so one call serves them all", func() {
			Expect(lowestResponseJailbreakThreshold([]config.ResponseJailbreakRule{
				{Name: "lenient", Threshold: 0.9},
				{Name: "strict", Threshold: 0.4},
			})).To(BeNumerically("~", 0.4, 1e-6))
		})
	})
})
