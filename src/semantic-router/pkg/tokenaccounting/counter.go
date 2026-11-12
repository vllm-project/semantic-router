package tokenaccounting

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Calibration returns an online-calibrated token count for textBytes of
// prompt text, and whether enough provider samples back it.
type Calibration func(textBytes int) (tokens int, calibrated bool)

// Count is a request's prompt token count and its provenance.
type Count struct {
	// Tokens is the prompt count, ReserveTokens included.
	Tokens int
	// ReserveTokens is the conservative part of Tokens that no text
	// tokenizer counts: images, audio, video and file payloads, opaque
	// reasoning state, and per-item chat framing.
	ReserveTokens int
	Source        Source
	// Family is the tokenizer family for SourceExact, else empty.
	Family string
}

// Counter counts requests against a Registry. Its methods are safe for
// concurrent use, and a nil Counter counts every request with the fallback.
type Counter struct {
	registry *Registry
	cache    *countCache
}

// Option configures a Counter.
type Option func(*Counter)

// WithCacheEntries bounds the per-text count cache; zero or less disables it.
func WithCacheEntries(entries int) Option {
	return func(counter *Counter) {
		counter.cache = newCountCache(entries)
	}
}

// NewCounter returns a Counter over registry, which may be nil.
func NewCounter(registry *Registry, options ...Option) *Counter {
	counter := &Counter{registry: registry, cache: newCountCache(DefaultCacheEntries)}
	for _, option := range options {
		option(counter)
	}
	return counter
}

// Count returns request's prompt tokens for model. A registered tokenizer
// gives SourceExact. Otherwise it is llmprotocol.EstimateInput: with its
// text-rate part replaced by calibration when calibration reports a
// calibrated count (SourceCalibrated), else unchanged (SourceFallback).
// Count never changes request.
func (counter *Counter) Count(model string, request *llmprotocol.Request, calibration Calibration) Count {
	if request == nil {
		return Count{Source: SourceFallback}
	}
	if counter != nil {
		if family, err := counter.registry.Resolve(model); err == nil {
			return counter.exact(family, request)
		}
	}
	return estimated(request, calibration)
}

// CacheStats reports the count cache's size and use.
func (counter *Counter) CacheStats() CacheStats {
	if counter == nil {
		return CacheStats{}
	}
	return counter.cache.stats()
}

func estimated(request *llmprotocol.Request, calibration Calibration) Count {
	estimate := llmprotocol.EstimateInput(request)
	count := Count{Tokens: estimate.Tokens, ReserveTokens: estimate.ReserveTokens, Source: SourceFallback}
	if calibration == nil || estimate.TextBytes <= 0 {
		return count
	}
	tokens, calibrated := calibration(estimate.TextBytes)
	if !calibrated || tokens <= 0 {
		return count
	}
	textRate := (estimate.TextBytes + llmprotocol.InputTextBytesPerToken - 1) / llmprotocol.InputTextBytesPerToken
	count.Tokens = llmprotocol.SaturatingTokenSum(estimate.Tokens-textRate, tokens)
	count.Source = SourceCalibrated
	return count
}

// exactWalk covers the components llmprotocol.EstimateInput covers, so an
// exact count and an estimate of one request describe the same prompt.
type exactWalk struct {
	counter *Counter
	family  *Family
	text    int
	reserve int
}

func (counter *Counter) exact(family *Family, request *llmprotocol.Request) Count {
	walk := exactWalk{counter: counter, family: family}
	walk.frame(len(request.Messages)+len(request.Instructions), llmprotocol.InputMessageFramingTokens)
	for _, instruction := range request.Instructions {
		walk.contents(instruction.Content)
	}
	for _, message := range request.Messages {
		walk.contents(message.Content)
	}
	for _, tool := range request.Tools {
		walk.count(tool.Name)
		walk.count(tool.Description)
		walk.count(string(tool.InputSchema))
		if tool.CustomFormat != nil {
			walk.count(tool.CustomFormat.Definition)
		}
		walk.frame(1, llmprotocol.InputToolDefinitionFramingTokens)
	}
	walk.count(request.OutputFormat.Name)
	walk.count(request.OutputFormat.Description)
	walk.count(string(request.OutputFormat.Schema))
	return Count{
		Tokens:        llmprotocol.SaturatingTokenSum(walk.text, walk.reserve),
		ReserveTokens: walk.reserve,
		Source:        SourceExact,
		Family:        family.Name,
	}
}

func (walk *exactWalk) contents(contents []llmprotocol.Content) {
	for _, content := range contents {
		switch content.Kind {
		case llmprotocol.ContentText, llmprotocol.ContentRefusal, llmprotocol.ContentReasoning:
			walk.count(content.Text)
			if detail := content.ReasoningDetail; detail != nil {
				walk.count(detail.Text.Value)
				walk.count(detail.Summary.Value)
				walk.count(detail.ToolName.Value)
				walk.count(detail.ToolCallID.Value)
				walk.count(detail.Arguments.Value)
				walk.count(detail.Result.Value)
				// Encrypted and signed state is opaque to a text tokenizer.
				walk.reserveBytes(len(detail.Data.Value), len(detail.Signature.Value))
			}
		case llmprotocol.ContentImage:
			walk.frame(1, llmprotocol.InputImageTokens)
		case llmprotocol.ContentAudio, llmprotocol.ContentVideo, llmprotocol.ContentFile:
			walk.reserveBytes(len(content.URL), len(content.Data), len(content.FileID))
		case llmprotocol.ContentToolCall:
			walk.frame(1, llmprotocol.InputToolCallFramingTokens)
			if content.ToolCall != nil {
				walk.count(content.ToolCall.Name)
				walk.count(content.ToolCall.Arguments)
			}
		case llmprotocol.ContentToolResult:
			if content.ToolResult != nil {
				walk.contents(content.ToolResult.Content)
			}
		}
	}
}

func (walk *exactWalk) count(text string) {
	if text == "" {
		return
	}
	walk.text = llmprotocol.SaturatingTokenSum(walk.text, walk.counter.cache.count(walk.family, text))
}

func (walk *exactWalk) frame(items, tokensEach int) {
	for range items {
		walk.reserve = llmprotocol.SaturatingTokenSum(walk.reserve, tokensEach)
	}
}

func (walk *exactWalk) reserveBytes(lengths ...int) {
	walk.reserve = llmprotocol.SaturatingTokenSum(append([]int{walk.reserve}, lengths...)...)
}
