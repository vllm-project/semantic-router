package extproc

import (
	"errors"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount"
)

// A chat template wraps each message, tool call and tool definition in a few
// tokens of its own and opens the answer with a few more. A count of the
// prompt's text adds these reserves for them, an approximation that the
// window's margin absorbs.
const (
	contextTemplateReserve       = 16
	contextMessageReserve        = llmprotocol.InputMessageFramingTokens
	contextToolCallReserve       = llmprotocol.InputToolCallFramingTokens
	contextToolDefinitionReserve = llmprotocol.InputToolDefinitionFramingTokens
)

var errContextMedia = errors.New("the request carries media, whose tokens no text tokenizer counts")

// contextCountKey names one count: a tokenizer's count of the request at one
// generation, as sent ("") or as a decision reshapes its prompt. The byte
// bound, which needs no tokenizer, leaves tokenizer empty.
type contextCountKey struct {
	tokenizer  string
	shape      string
	generation uint64
}

type contextCount struct {
	tokens int
	ok     bool
}

// contextByteBound bounds the request's prompt tokens without a tokenizer:
// byte-level tokenizers never emit more tokens than the text has UTF-8 bytes,
// and the reserves cover chat framing and media.
func contextByteBound(request *llmprotocol.Request, decision *config.Decision) int {
	if request == nil {
		return 0
	}
	tokens, _ := overflowTokenCounter{decision: decision}.CountRequest("", &contextcompression.RequestIR{Semantic: request})
	return tokens
}

// requestContextByteBound is contextByteBound of the request in flight, kept
// for its generation and prompt shape.
func requestContextByteBound(ctx *RequestContext, decision *config.Decision) int {
	key := contextCountKey{shape: promptShape(decision), generation: ctx.SemanticRequest.Generation}
	if bound, cached := ctx.contextBounds[key]; cached {
		return bound
	}
	bound := contextByteBound(ctx.SemanticRequest, decision)
	if ctx.contextBounds == nil {
		ctx.contextBounds = make(map[contextCountKey]int)
	}
	ctx.contextBounds[key] = bound
	return bound
}

// promptShape is "" when a decision dispatches the prompt as sent, else the
// decision's name: its system prompt or tool policy changes what a backend
// counts, so its count is its own.
func promptShape(decision *config.Decision) string {
	if decision == nil {
		return ""
	}
	if decision.GetSystemPromptConfig() != nil && decision.IsSystemPromptEnabled() {
		return decision.Name
	}
	if decision.GetToolsConfig().ActivePolicy().EffectiveMode() == config.ToolsPolicyModeNone {
		return decision.Name
	}
	return ""
}

// contextCountKind says which tokenizer counted a model's prompt: the one
// the model names (exact), or the proxy tokenizer, whose count can be well
// off the model's own and gets the wider proxy_margin band.
type contextCountKind string

const (
	contextCountExact contextCountKind = "exact"
	contextCountProxy contextCountKind = "proxy"
)

// modelTokenizer is the tokenizer that counts a model's prompts, and which
// kind of count it gives.
type modelTokenizer struct {
	encoding *tokencount.Encoding
	kind     contextCountKind
}

// promptTokenizer is the tokenizer that counts model's prompts: the one its
// configuration names, once loaded, else the proxy tokenizer.
func (r *OpenAIRouter) promptTokenizer(model string) modelTokenizer {
	registry := r.tokenizers
	if registry == nil {
		registry = tokencount.Default()
	}
	if tokenizer := r.Config.ModelConfig[strings.TrimSpace(model)].Tokenizer; tokenizer != "" {
		if encoding, ok := registry.Lookup(tokenizer); ok {
			return modelTokenizer{encoding: encoding, kind: contextCountExact}
		}
	}
	if encoding, ok := registry.Lookup(r.Config.ContextWindow.EffectiveProxyTokenizer()); ok {
		return modelTokenizer{encoding: encoding, kind: contextCountProxy}
	}
	encoding, _ := tokencount.Builtin(config.DefaultProxyTokenizer)
	return modelTokenizer{encoding: encoding, kind: contextCountProxy}
}

// requestTokenizer is promptTokenizer, fixed for the rest of the request at
// its first answer, so that a tokenizer that finishes loading mid-request
// does not change one request's counts.
func (r *OpenAIRouter) requestTokenizer(ctx *RequestContext, model string) modelTokenizer {
	model = strings.TrimSpace(model)
	if tokenizer, ok := ctx.contextTokenizers[model]; ok {
		return tokenizer
	}
	tokenizer := r.promptTokenizer(model)
	if ctx.contextTokenizers == nil {
		ctx.contextTokenizers = map[string]modelTokenizer{}
	}
	ctx.contextTokenizers[model] = tokenizer
	return tokenizer
}

// prepareContextWindowCheck starts loading every tokenizer the configuration
// names, so that requests find them loaded, and publishes the bands.
func prepareContextWindowCheck(cfg *config.RouterConfig) {
	if cfg == nil {
		return
	}
	tokenizers := []string{cfg.ContextWindow.EffectiveProxyTokenizer()}
	for _, params := range cfg.ModelConfig {
		tokenizers = append(tokenizers, params.Tokenizer)
	}
	tokencount.Default().Preload(tokenizers...)
	metrics.SetContextWindowMargins(cfg.ContextWindow.EffectiveMargin(), cfg.ContextWindow.EffectiveProxyMargin())
}

// contextTokens is the request's prompt token count by model's tokenizer,
// counted in the Router once per tokenizer and prompt shape, and the kind of
// that count. ok is false for a request that carries media, whose tokens no
// text tokenizer counts; the window check never refuses such a request.
func (r *OpenAIRouter) contextTokens(ctx *RequestContext, decision *config.Decision, model string) (int, contextCountKind, bool) {
	if r == nil || r.Config == nil || ctx == nil || ctx.SemanticRequest == nil {
		return 0, "", false
	}
	tokenizer := r.requestTokenizer(ctx, model)
	if tokenizer.encoding == nil {
		return 0, "", false
	}
	key := contextCountKey{tokenizer: tokenizer.encoding.Name(), shape: promptShape(decision), generation: ctx.SemanticRequest.Generation}
	if count, cached := ctx.contextCounts[key]; cached {
		return count.tokens, tokenizer.kind, count.ok
	}
	started := time.Now()
	request := ctx.SemanticRequest
	var err error
	switch {
	case requestCarriesMedia(request):
		err = errContextMedia
	case key.shape != "":
		request, err = selection.EffectiveCandidateRequest(request, decision)
	}
	count := contextCount{}
	if err == nil {
		count = contextCount{tokens: countPromptTokens(tokenizer.encoding, request), ok: true}
	}
	if ctx.contextCounts == nil {
		ctx.contextCounts = make(map[contextCountKey]contextCount)
	}
	ctx.contextCounts[key] = count
	logContextCount(ctx, key, count, err, time.Since(started))
	return count.tokens, tokenizer.kind, count.ok
}

// cachedContextTokens is a count contextTokens already holds for model, and
// its kind.
func (r *OpenAIRouter) cachedContextTokens(ctx *RequestContext, decision *config.Decision, model string) (int, contextCountKind, bool) {
	if r == nil || r.Config == nil || ctx == nil || ctx.SemanticRequest == nil {
		return 0, "", false
	}
	tokenizer := r.requestTokenizer(ctx, model)
	if tokenizer.encoding == nil {
		return 0, "", false
	}
	count, cached := ctx.contextCounts[contextCountKey{
		tokenizer: tokenizer.encoding.Name(), shape: promptShape(decision), generation: ctx.SemanticRequest.Generation,
	}]
	return count.tokens, tokenizer.kind, cached && count.ok
}

func logContextCount(ctx *RequestContext, key contextCountKey, count contextCount, err error, elapsed time.Duration) {
	fields := map[string]interface{}{
		"request_id": ctx.RequestID,
		"tokenizer":  key.tokenizer,
		"elapsed_ms": elapsed.Milliseconds(),
	}
	if key.shape != "" {
		fields["decision"] = key.shape
	}
	if err != nil {
		fields["reason"] = err.Error()
		logging.ComponentDebugEvent("extproc", "context_tokens_not_counted", fields)
		return
	}
	fields["tokens"] = count.tokens
	logging.ComponentEvent("extproc", "context_tokens_counted", fields)
}

// countPromptTokens counts what a chat template renders of request: its
// text, tool calls and results, tool definitions and output schema, with the
// template's reserves.
func countPromptTokens(tokenizer *tokencount.Encoding, request *llmprotocol.Request) int {
	var text strings.Builder
	reserve := contextTemplateReserve
	write := func(parts ...string) {
		for _, part := range parts {
			if part != "" {
				text.WriteString(part)
				text.WriteByte('\n')
			}
		}
	}
	var consume func([]llmprotocol.Content)
	consume = func(contents []llmprotocol.Content) {
		for _, content := range contents {
			switch content.Kind {
			case llmprotocol.ContentText, llmprotocol.ContentRefusal, llmprotocol.ContentReasoning:
				write(content.Text)
				if detail := content.ReasoningDetail; detail != nil {
					write(detail.Text.Value, detail.Summary.Value)
				}
			case llmprotocol.ContentToolCall:
				reserve = llmprotocol.SaturatingTokenSum(reserve, contextToolCallReserve)
				if content.ToolCall != nil {
					write(content.ToolCall.Name, content.ToolCall.Arguments)
				}
			case llmprotocol.ContentToolResult:
				if content.ToolResult != nil {
					consume(content.ToolResult.Content)
				}
			}
		}
	}
	for _, instruction := range request.Instructions {
		reserve = llmprotocol.SaturatingTokenSum(reserve, contextMessageReserve)
		consume(instruction.Content)
	}
	for _, message := range request.Messages {
		reserve = llmprotocol.SaturatingTokenSum(reserve, contextMessageReserve)
		consume(message.Content)
	}
	for _, tool := range request.Tools {
		reserve = llmprotocol.SaturatingTokenSum(reserve, contextToolDefinitionReserve)
		write(tool.Name, tool.Description, string(tool.InputSchema))
		if tool.CustomFormat != nil {
			write(tool.CustomFormat.Definition)
		}
	}
	write(string(request.OutputFormat.Schema), request.OutputFormat.Name, request.OutputFormat.Description)
	return llmprotocol.SaturatingTokenSum(tokenizer.Count(text.String()), reserve)
}

// requestCarriesMedia reports images, audio, video or files anywhere in the
// prompt, whose tokens a text tokenizer leaves out of the count.
func requestCarriesMedia(request *llmprotocol.Request) bool {
	if request == nil {
		return false
	}
	var carries func([]llmprotocol.Content) bool
	carries = func(contents []llmprotocol.Content) bool {
		for _, content := range contents {
			switch content.Kind {
			case llmprotocol.ContentImage, llmprotocol.ContentAudio, llmprotocol.ContentVideo, llmprotocol.ContentFile:
				return true
			case llmprotocol.ContentToolResult:
				if content.ToolResult != nil && carries(content.ToolResult.Content) {
					return true
				}
			}
		}
		return false
	}
	for _, instruction := range request.Instructions {
		if carries(instruction.Content) {
			return true
		}
	}
	for _, message := range request.Messages {
		if carries(message.Content) {
			return true
		}
	}
	return false
}
