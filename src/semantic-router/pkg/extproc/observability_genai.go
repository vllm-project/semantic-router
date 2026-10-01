package extproc

import (
	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.41.0"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// Catalog provider IDs that have a well-known gen_ai.provider.name. Any other
// ID is reported unchanged, which the convention allows as a custom value.
var genAIProviderNames = map[string]attribute.KeyValue{
	"anthropic":    semconv.GenAIProviderNameAnthropic,
	"azure-openai": semconv.GenAIProviderNameAzureAIOpenAI,
	"bedrock":      semconv.GenAIProviderNameAWSBedrock,
	"cohere":       semconv.GenAIProviderNameCohere,
	"deepseek":     semconv.GenAIProviderNameDeepseek,
	"gemini":       semconv.GenAIProviderNameGCPGemini,
	"groq":         semconv.GenAIProviderNameGroq,
	"mistral":      semconv.GenAIProviderNameMistralAI,
	"openai":       semconv.GenAIProviderNameOpenAI,
	"perplexity":   semconv.GenAIProviderNamePerplexity,
	"vertex-ai":    semconv.GenAIProviderNameGCPVertexAI,
	"xai":          semconv.GenAIProviderNameXAI,
	// Well-known only after semconv v1.41.0: https://github.com/open-telemetry/semantic-conventions-genai/blob/e57c543b4889619eb2a05702471937db5119165d/docs/registry/attributes/gen-ai.md#gen-ai-provider-name
	"moonshot": semconv.GenAIProviderNameKey.String("moonshot_ai"),
}

// The Images wire has no matching well-known operation name, so image
// dispatches carry no GenAI attributes.
func genAIOperation(format llmprotocol.WireFormat) (attribute.KeyValue, bool) {
	switch format {
	case llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1:
		return semconv.GenAIOperationNameChat, true
	default:
		return attribute.KeyValue{}, false
	}
}

func genAIRequestAttributes(dispatch *providerDispatch) []attribute.KeyValue {
	operation, ok := genAIOperation(dispatch.targetFormat)
	if !ok {
		return nil
	}
	attrs := []attribute.KeyValue{operation}
	if provider, known := genAIProviderName(dispatch.profile); known {
		attrs = append(attrs, provider)
	}
	if dispatch.upstreamModel != "" {
		attrs = append(attrs, semconv.GenAIRequestModel(dispatch.upstreamModel))
	}
	return attrs
}

func genAIProviderName(profile *config.ProviderProfile) (attribute.KeyValue, bool) {
	if profile == nil {
		return attribute.KeyValue{}, false
	}
	providerID, err := profile.ProviderType()
	if err != nil {
		return attribute.KeyValue{}, false
	}
	if name, ok := genAIProviderNames[providerID]; ok {
		return name, true
	}
	return semconv.GenAIProviderNameKey.String(providerID), true
}

// observeUpstreamResponse records what the provider reported. Usage that is
// missing or not authoritative is left off rather than estimated.
func observeUpstreamResponse(ctx *RequestContext, responseModel string, usage responseUsageMetrics) {
	if ctx == nil || ctx.UpstreamSpan == nil {
		return
	}
	if _, ok := genAIOperation(ctx.TargetFormat); !ok {
		return
	}
	var attrs []attribute.KeyValue
	if responseModel != "" {
		attrs = append(attrs, semconv.GenAIResponseModel(responseModel))
	}
	if !usage.invalid {
		attrs = append(attrs,
			semconv.GenAIUsageInputTokens(usage.promptTokens),
			semconv.GenAIUsageOutputTokens(usage.completionTokens),
		)
	}
	tracing.SetSpanAttributes(ctx.UpstreamSpan, attrs...)
}
