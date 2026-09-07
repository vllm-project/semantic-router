package extproc

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var _ = Describe("Response stage jailbreak signal", func() {
	// The rules are read from the recipe the request resolved to, through its
	// classifier, so the router gets a real single-profile classifier graph
	// and the request context a resolved default recipe. prompt_guard stays
	// disabled: the rule is declared but no detector backs it.
	newRouter := func(rules ...config.JailbreakRule) (*OpenAIRouter, *RequestContext) {
		cfg := &config.RouterConfig{}
		cfg.Signals.JailbreakRules = rules
		classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
		Expect(err).NotTo(HaveOccurred())
		router := &OpenAIRouter{Config: cfg, Classifier: classifiers.Default(), RecipeClassifiers: classifiers}
		ctx := &RequestContext{} // no VSRSelectedDecision, so no plugin at all
		ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
		return router, ctx
	}
	responseRule := config.JailbreakRule{Name: "unsafe_completion", Threshold: 0.7, Direction: config.SignalDirectionResponse}

	// The signal is driven by the declared rules, not by an enforcement plugin.
	// It has to be published whether or not response_jailbreak is enabled
	// anywhere, or it is plugin state wearing a signal's name.
	It("publishes without any plugin configured", func() {
		router, ctx := newRouter(responseRule)

		router.evaluateResponseJailbreakSignal(ctx, "some answer")

		// No detector backs the rule, so it resolves as unavailable rather
		// than silently absent, under the plain jailbreak key.
		Expect(ctx.VSRSignalErrors).To(HaveKey("jailbreak:unsafe_completion"))
		Expect(ctx.VSRMatchedResponseJailbreak).To(BeEmpty())
	})

	It("does nothing when only request-direction rules are declared", func() {
		router, ctx := newRouter(config.JailbreakRule{Name: "prompt_injection", Threshold: 0.7})

		router.evaluateResponseJailbreakSignal(ctx, "some answer")

		Expect(ctx.VSRSignalErrors).To(BeEmpty())
		Expect(ctx.VSRSignalConfidences).To(BeEmpty())
	})

	It("does nothing for a request that resolved no recipe", func() {
		router, _ := newRouter(responseRule)
		ctx := &RequestContext{}

		router.evaluateResponseJailbreakSignal(ctx, "some answer")

		Expect(ctx.VSRSignalErrors).To(BeEmpty())
		Expect(ctx.VSRSignalConfidences).To(BeEmpty())
	})

	It("reports a response with no assistant text as unresolved, not clean", func() {
		router, ctx := newRouter(responseRule)

		router.evaluateResponseJailbreakSignal(ctx, "")

		Expect(ctx.VSRSignalErrors).To(HaveKey("jailbreak:unsafe_completion"))
	})

	It("reads the outcome from the response-direction rules only", func() {
		router, ctx := newRouter(responseRule, config.JailbreakRule{Name: "prompt_injection", Threshold: 0.7})
		// A failed request-stage scan under the same type must not make the
		// response scan look unresolved.
		ctx.VSRSignalErrors = map[string]string{"jailbreak:prompt_injection": "jailbreak_evaluation_failed"}
		ctx.VSRMatchedResponseJailbreak = []string{"unsafe_completion"}

		matched, resolved := router.responseJailbreakSignalOutcome(ctx)
		Expect(resolved).To(BeTrue())
		Expect(matched).To(BeTrue())
	})

	// A rule that matched is enforced even when a stricter rule was left
	// unresolved by the same partial scan: reporting the response as
	// unresolved would send a real detection through the on_error policy.
	It("reports a match even when another rule is unresolved", func() {
		router, ctx := newRouter(responseRule, config.JailbreakRule{Name: "strict", Threshold: 0.95, Direction: config.SignalDirectionResponse})
		ctx.VSRSignalErrors = map[string]string{"jailbreak:strict": "response_jailbreak_evaluation_failed"}
		ctx.VSRMatchedResponseJailbreak = []string{"unsafe_completion"}

		matched, resolved := router.responseJailbreakSignalOutcome(ctx)
		Expect(matched).To(BeTrue())
		Expect(resolved).To(BeTrue())
	})
})
