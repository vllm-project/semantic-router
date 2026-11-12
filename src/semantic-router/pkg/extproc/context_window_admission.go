package extproc

import (
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// contextWindowVerdict is whether a request's prompt fits one model's
// context window, and on what evidence.
type contextWindowVerdict int

const (
	// contextFits: the byte bound fits, or the Router's count is at most the
	// window less its margin, or there is no count and the routing estimate
	// fits.
	contextFits contextWindowVerdict = iota
	// contextInBand: the count is within the margin of the window. The
	// request goes on, and the backend decides.
	contextInBand
	// contextExceeds: the count is at least the window plus its margin.
	// Only this refuses.
	contextExceeds
	// contextLikelyExceeds: there is no count, and the routing estimate does
	// not fit.
	contextLikelyExceeds
)

// excludes reports a verdict that takes a model out of a candidate set.
func (v contextWindowVerdict) excludes() bool {
	return v == contextExceeds || v == contextLikelyExceeds
}

func (v contextWindowVerdict) String() string {
	switch v {
	case contextInBand:
		return "in_band"
	case contextExceeds:
		return "exceeds"
	case contextLikelyExceeds:
		return "likely_exceeds"
	}
	return "fits"
}

// contextLengthError refuses a request for its length on the Router's count:
// the widest window that could serve it, the request's estimated count
// there, and the kind of that count.
type contextLengthError struct {
	window int
	tokens int
	kind   contextCountKind
	scope  string
}

func (e *contextLengthError) Error() string {
	return fmt.Sprintf("%s: %s has a %d-token context window and the request counts about %d tokens (%s count)",
		errNoContextEligibleDecisionModel, e.scope, e.window, e.tokens, e.kind)
}

func (e *contextLengthError) Unwrap() error { return errNoContextEligibleDecisionModel }

// contextLengthFailure is the client's refusal for e in the client's own
// protocol: OpenAI's context_length_exceeded, with a message that calls the
// count an estimate, or the message Anthropic's API gives a prompt that is
// too long, which clients parse as it is.
func contextLengthFailure(e *contextLengthError, format llmprotocol.WireFormat) routingFailure {
	failure := routingFailureContextLength
	if e == nil || e.window <= 0 {
		return failure
	}
	if format == llmprotocol.AnthropicMessagesV1 {
		failure.message = fmt.Sprintf("prompt is too long: %d tokens > %d maximum", e.tokens, e.window)
	} else {
		failure.message = fmt.Sprintf(
			"This model's maximum context length is %d tokens. However, your request has an estimated %d input tokens. "+
				"Please reduce the length of the input.", e.window, e.tokens)
	}
	return failure
}

func (r *OpenAIRouter) contextWindowSize(model string) int {
	if r == nil || r.Config == nil {
		return 0
	}
	return r.Config.ModelConfig[strings.TrimSpace(model)].ContextWindowSize
}

// contextWindowVerdict judges model's window for the request. The byte bound
// and the Router's count decide; estimate, the routing estimate, only orders
// candidates where there is no count, and never refuses. A model the window
// filter already kept against the estimate fits.
func (r *OpenAIRouter) contextWindowVerdict(
	ctx *RequestContext, decision *config.Decision, model string, estimate int,
) (contextWindowVerdict, int) {
	window := r.contextWindowSize(model)
	if window <= 0 {
		return contextFits, 0
	}
	if ctx != nil && ctx.SemanticRequest != nil {
		bound := requestContextByteBound(ctx, decision)
		if bound < window {
			return contextFits, 0
		}
		verdict, tokens, kind, decided := r.addedPromptVerdict(ctx, decision, model, window, bound)
		if !decided {
			var ok bool
			if tokens, kind, ok = r.contextTokens(ctx, decision, model); ok {
				verdict, decided = r.bandVerdict(window, tokens, kind), true
			}
		}
		if decided {
			if verdict == contextInBand {
				markContextInBand(ctx, model)
			}
			r.noteContextVerdict(ctx, decision, model, window, tokens, kind, verdict)
			return verdict, tokens
		}
	}
	if ctx != nil {
		if _, kept := ctx.contextWindowKept[strings.TrimSpace(model)]; kept {
			return contextFits, 0
		}
	}
	if estimate > 0 && estimate > window {
		return contextLikelyExceeds, 0
	}
	return contextFits, 0
}

// addedPromptVerdict judges a decision that only adds to the prompt, such as
// an inserted system prompt, from the count of the prompt as sent, without
// another count: that count is a lower bound, and it plus the bytes the
// decision adds an upper one.
func (r *OpenAIRouter) addedPromptVerdict(
	ctx *RequestContext, decision *config.Decision, model string, window, bound int,
) (contextWindowVerdict, int, contextCountKind, bool) {
	if promptShape(decision) == "" || decisionMayShrinkPrompt(decision) {
		return contextFits, 0, "", false
	}
	sent, kind, ok := r.cachedContextTokens(ctx, nil, model)
	if !ok {
		return contextFits, 0, "", false
	}
	if r.bandVerdict(window, sent, kind) == contextExceeds {
		return contextExceeds, sent, kind, true
	}
	if added := bound - requestContextByteBound(ctx, nil); r.bandVerdict(window, sent+added, kind) == contextFits {
		return contextFits, sent, kind, true
	}
	return contextFits, 0, "", false
}

// contextMargin is the band around a window for a count of kind: margin
// for the model's own tokenizer, proxy_margin for the proxy tokenizer.
func (r *OpenAIRouter) contextMargin(kind contextCountKind) float64 {
	if kind == contextCountExact {
		return r.Config.ContextWindow.EffectiveMargin()
	}
	return r.Config.ContextWindow.EffectiveProxyMargin()
}

// bandVerdict places a count of kind against a window and the band around
// it.
func (r *OpenAIRouter) bandVerdict(window, tokens int, kind contextCountKind) contextWindowVerdict {
	margin := r.contextMargin(kind)
	switch {
	case float64(tokens) <= float64(window)*(1-margin):
		return contextFits
	case float64(tokens) >= float64(window)*(1+margin):
		return contextExceeds
	}
	return contextInBand
}

// noteContextVerdict logs and counts, once per model and prompt shape of a
// request, how the Router's count placed the request against the model's
// window: the kind of count, the band it used, and the verdict.
func (r *OpenAIRouter) noteContextVerdict(
	ctx *RequestContext, decision *config.Decision, model string, window, tokens int,
	kind contextCountKind, verdict contextWindowVerdict,
) {
	key := strings.TrimSpace(model) + "\x00" + promptShape(decision)
	if _, noted := ctx.contextVerdicts[key]; noted {
		return
	}
	if ctx.contextVerdicts == nil {
		ctx.contextVerdicts = map[string]struct{}{}
	}
	ctx.contextVerdicts[key] = struct{}{}
	metrics.RecordContextWindowCheck(string(kind), verdict.String())
	fields := map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      strings.TrimSpace(model),
		"window":     window,
		"tokens":     tokens,
		"count":      string(kind),
		"margin":     r.contextMargin(kind),
		"verdict":    verdict.String(),
	}
	if shape := promptShape(decision); shape != "" {
		fields["decision"] = shape
	}
	logging.ComponentEvent("extproc", "context_window_checked", fields)
}

// markContextInBand notes a model whose window the request's count falls
// in the margin of, whose backend may refuse the request for its length.
func markContextInBand(ctx *RequestContext, model string) {
	if ctx.contextInBand == nil {
		ctx.contextInBand = map[string]struct{}{}
	}
	ctx.contextInBand[strings.TrimSpace(model)] = struct{}{}
}

// modelExceedsContextWindow reports whether model should leave a candidate
// set for the request's length, on the count or on the estimate.
func (r *OpenAIRouter) modelExceedsContextWindow(ctx *RequestContext, model string, estimate int) bool {
	verdict, _ := r.contextWindowVerdict(ctx, contextDecision(ctx), model, estimate)
	return verdict.excludes()
}

func contextDecision(ctx *RequestContext) *config.Decision {
	if ctx == nil {
		return nil
	}
	return ctx.VSRSelectedDecision
}

// keepUnprovenContextModels returns the refs only the estimate excluded that
// a decision keeps when the refs that fit fall short of its minimum: the
// widest first, and every ref as wide as the widest kept, until the minimum
// is met. The backend then decides, and later window checks of the request
// treat the kept models as fitting.
func (r *OpenAIRouter) keepUnprovenContextModels(
	ctx *RequestContext, refs, eligible []config.ModelRef, contextTokens, minimum int, decisionName string,
) []config.ModelRef {
	need := minimum - distinctModelRefCount(eligible)
	if need <= 0 {
		return eligible
	}
	var unproven []config.ModelRef
	for _, ref := range refs {
		if modelRefInEligibility(ref, eligible) {
			continue
		}
		if verdict, _ := r.contextWindowVerdict(ctx, contextDecision(ctx), ref.Model, contextTokens); verdict == contextLikelyExceeds {
			unproven = append(unproven, ref)
		}
	}
	if len(unproven) == 0 {
		return eligible
	}
	sort.SliceStable(unproven, func(i, j int) bool {
		return r.contextWindowSize(unproven[i].Model) > r.contextWindowSize(unproven[j].Model)
	})
	kept := map[string]struct{}{}
	widest := 0
	for _, ref := range unproven {
		window := r.contextWindowSize(ref.Model)
		if len(kept) >= need && window < widest {
			break
		}
		widest = max(widest, window)
		kept[strings.TrimSpace(ref.Model)] = struct{}{}
	}
	if ctx.contextWindowKept == nil {
		ctx.contextWindowKept = map[string]struct{}{}
	}
	result := make([]config.ModelRef, 0, len(eligible)+len(kept))
	names := make([]string, 0, len(kept))
	for _, ref := range refs {
		model := strings.TrimSpace(ref.Model)
		if _, ok := kept[model]; ok {
			ctx.contextWindowKept[model] = struct{}{}
			names = append(names, model)
			result = append(result, ref)
		} else if modelRefInEligibility(ref, eligible) {
			result = append(result, ref)
		}
	}
	logging.ComponentEvent("extproc", "context_window_unproven", map[string]interface{}{
		"request_id":     ctx.RequestID,
		"decision":       decisionName,
		"context_tokens": contextTokens,
		"kept":           names,
	})
	return result
}

// contextLengthRefusal is the refusal for models whose counts all exceed
// their windows beyond the margin, naming the widest of them; nil when one of
// them is not ruled out that way.
func (r *OpenAIRouter) contextLengthRefusal(ctx *RequestContext, models []string, contextTokens int, scope string) *contextLengthError {
	var refusal *contextLengthError
	for _, model := range models {
		verdict, tokens := r.contextWindowVerdict(ctx, contextDecision(ctx), model, contextTokens)
		if verdict != contextExceeds {
			return nil
		}
		if window := r.contextWindowSize(model); refusal == nil || window > refusal.window {
			refusal = &contextLengthError{window: window, tokens: tokens, kind: r.requestTokenizer(ctx, model).kind, scope: scope}
		}
	}
	return refusal
}

func modelRefNames(refs []config.ModelRef) []string {
	names := make([]string, 0, len(refs))
	for _, ref := range refs {
		names = append(names, ref.Model)
	}
	return names
}

// refuseForLength is contextLengthFailure for a refusal the client gets,
// counted in llm_context_window_refusals_total.
func refuseForLength(refusal *contextLengthError, format llmprotocol.WireFormat) routingFailure {
	if refusal != nil {
		metrics.RecordContextWindowRefusal(string(refusal.kind))
	}
	return contextLengthFailure(refusal, format)
}

// promptShrink is how much smaller than the client's prompt the decisions a
// request can reach may dispatch it.
type promptShrink struct {
	// unbounded: a decision may compress the request.
	unbounded bool
	// instructions and tools: a decision may drop the client's system and
	// developer instructions, or its tools and tool history.
	instructions, tools bool
}

func (s *promptShrink) add(decision *config.Decision) {
	s.unbounded = s.unbounded || contextCompression(decision.Plugins) != nil
	s.instructions = s.instructions || (decision.GetSystemPromptConfig() != nil && decision.IsSystemPromptEnabled() &&
		decision.GetSystemPromptMode() != config.SystemPromptModeInsert)
	s.tools = s.tools || decision.GetToolsConfig().ActivePolicy().EffectiveMode() == config.ToolsPolicyModeNone
}

// removableBytes bounds the tokens of what the decisions may drop: a
// byte-level tokenizer counts no more tokens than the bytes.
func (s promptShrink) removableBytes(request *llmprotocol.Request) int {
	removable := 0
	if s.instructions {
		instructions := llmprotocol.EstimateInput(&llmprotocol.Request{Instructions: request.Instructions})
		removable = llmprotocol.SaturatingTokenSum(removable, instructions.TextBytes, instructions.StructuredBytes)
	}
	if s.tools {
		tools := llmprotocol.EstimateInput(&llmprotocol.Request{Tools: request.Tools})
		removable = llmprotocol.SaturatingTokenSum(removable, tools.TextBytes, tools.StructuredBytes)
		for _, message := range request.Messages {
			for _, content := range message.Content {
				if content.Kind != llmprotocol.ContentToolCall && content.Kind != llmprotocol.ContentToolResult {
					continue
				}
				part := llmprotocol.EstimateInput(&llmprotocol.Request{Messages: []llmprotocol.Message{{Content: []llmprotocol.Content{content}}}})
				removable = llmprotocol.SaturatingTokenSum(removable, part.TextBytes, part.StructuredBytes)
			}
		}
	}
	return removable
}

// reachableContextModels are the models a request can be dispatched to: the
// one it names, or for an entrypoint every model its recipe's decisions and
// default can choose, with how much the recipe's decisions may shrink the
// prompt before dispatch.
func (r *OpenAIRouter) reachableContextModels(originalModel string, ctx *RequestContext) ([]string, promptShrink) {
	var shrink promptShrink
	if r == nil || r.Config == nil {
		return nil, shrink
	}
	if !r.requestModelActsAsAuto(originalModel) {
		if r.Config.RoutesModelToBackend(originalModel) && !r.usesExternalGatewayDispatch(originalModel) {
			return []string{strings.TrimSpace(originalModel)}, shrink
		}
		return nil, shrink
	}
	recipe := ctx.Routing.SelectedRecipe()
	if recipe == nil {
		return nil, shrink
	}
	var models []string
	add := func(model string) {
		if model = strings.TrimSpace(model); model != "" && !slices.Contains(models, model) {
			models = append(models, model)
		}
	}
	shrink.unbounded = contextCompression(recipe.Profile.DefaultPlugins) != nil
	for i := range recipe.Profile.Decisions {
		decision := &recipe.Profile.Decisions[i]
		if decision.GetFastResponseConfig() != nil {
			continue
		}
		shrink.add(decision)
		for _, ref := range decision.ModelRefs {
			add(ref.Model)
		}
		if decision.Action != nil && decision.Action.Type == config.DecisionActionRoute {
			add(decision.Action.Destination)
		}
		for _, model := range explicitAlgorithmModels(decision.Algorithm) {
			add(model)
		}
	}
	add(r.Config.DefaultModel)
	return models, shrink
}

// decisionMayShrinkPrompt reports a decision that can dispatch a smaller
// prompt than the client sent: context compression, a system prompt that
// replaces the client's, or a tool policy that drops tools.
func decisionMayShrinkPrompt(decision *config.Decision) bool {
	var shrink promptShrink
	shrink.add(decision)
	return shrink.unbounded || shrink.instructions || shrink.tools
}
