package extproc

import (
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type openAIBackendDialectKind string

const (
	openAIBackendDialectOfficialOpenAI      openAIBackendDialectKind = "official_openai"
	openAIBackendDialectOfficialDeepSeek    openAIBackendDialectKind = "official_deepseek"
	openAIBackendDialectOpenRouter          openAIBackendDialectKind = "openrouter"
	openAIBackendDialectVLLM                openAIBackendDialectKind = "vllm"
	openAIBackendDialectAzureOpenAI         openAIBackendDialectKind = "azure_openai"
	openAIBackendDialectGenericOpenAICompat openAIBackendDialectKind = "generic_openai_compatible"
)

type openAIBackendDialect struct {
	kind                            openAIBackendDialectKind
	supportsTopLevelReasoningEffort bool
	supportsTopLevelDeepSeekThink   bool
	vendorExtensions                llmprotocol.ResponseVendor
}

// resolveOpenAIBackendDialect captures request-shaping differences between
// OpenAI-compatible backends. A nil profile is the legacy local-vLLM path, where
// reasoning_effort must stay in chat_template_kwargs.
func resolveOpenAIBackendDialect(profile *config.ProviderProfile) openAIBackendDialect {
	if profile == nil {
		return newOpenAIBackendDialect(openAIBackendDialectVLLM)
	}
	if strings.EqualFold(profile.Type, "azure-openai") {
		return newOpenAIBackendDialect(openAIBackendDialectAzureOpenAI)
	}
	if !strings.EqualFold(profile.Type, "openai") {
		return newOpenAIBackendDialect(openAIBackendDialectGenericOpenAICompat)
	}

	switch normalizedProfileHost(profile) {
	case "api.openai.com":
		return newOpenAIBackendDialect(openAIBackendDialectOfficialOpenAI)
	case "api.deepseek.com":
		// DeepSeek's official OpenAI-compatible API accepts top-level
		// reasoning_effort and uses top-level thinking for reasoning on/off.
		return newOpenAIBackendDialect(openAIBackendDialectOfficialDeepSeek)
	case "openrouter.ai":
		// OpenRouter exposes reasoning_effort as a top-level OpenAI-compatible
		// request field; local vLLM-compatible endpoints do not.
		return newOpenAIBackendDialect(openAIBackendDialectOpenRouter)
	}

	if isAzureOpenAIHost(normalizedProfileHost(profile)) {
		return newOpenAIBackendDialect(openAIBackendDialectAzureOpenAI)
	}
	return newOpenAIBackendDialect(openAIBackendDialectGenericOpenAICompat)
}

func isAzureOpenAIHost(host string) bool {
	return host != "" &&
		(strings.HasSuffix(host, ".openai.azure.com") ||
			strings.HasSuffix(host, ".services.ai.azure.com") ||
			strings.HasSuffix(host, ".cognitiveservices.azure.com"))
}

func newOpenAIBackendDialect(kind openAIBackendDialectKind) openAIBackendDialect {
	dialect := openAIBackendDialect{
		kind: kind,
	}
	switch kind {
	case openAIBackendDialectOfficialOpenAI:
		dialect.supportsTopLevelReasoningEffort = true
	case openAIBackendDialectOfficialDeepSeek:
		dialect.supportsTopLevelReasoningEffort = true
		dialect.supportsTopLevelDeepSeekThink = true
	case openAIBackendDialectOpenRouter:
		dialect.supportsTopLevelReasoningEffort = true
	case openAIBackendDialectAzureOpenAI:
		dialect.vendorExtensions = llmprotocol.ResponseVendorAzure
	}
	return dialect
}

func (d openAIBackendDialect) vendorExtensionProvider() llmprotocol.ResponseVendor {
	return d.vendorExtensions
}

func (d openAIBackendDialect) usesTopLevelReasoningEffort() bool {
	return d.supportsTopLevelReasoningEffort
}

func (d openAIBackendDialect) usesDeepSeekOfficialReasoning() bool {
	return d.kind == openAIBackendDialectOfficialDeepSeek && d.supportsTopLevelDeepSeekThink
}

func normalizedProfileHost(profile *config.ProviderProfile) string {
	if profile == nil || profile.BaseURL == "" {
		return ""
	}
	u, err := url.Parse(profile.BaseURL)
	if err != nil {
		return ""
	}
	return strings.ToLower(u.Hostname())
}
