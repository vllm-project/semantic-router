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

// azureOpenAIHostSuffixes are the host suffixes Azure serves its
// OpenAI-compatible endpoints from. Azure resources are per-tenant subdomains,
// so unlike the other dialects it is matched by suffix rather than exact host.
var azureOpenAIHostSuffixes = []string{
	".openai.azure.com",
	".services.ai.azure.com",
	".cognitiveservices.azure.com",
}

type openAIBackendDialect struct {
	kind                            openAIBackendDialectKind
	supportsTopLevelReasoningEffort bool
	supportsTopLevelDeepSeekThink   bool
	// vendorExtensions names the provider whose documented response
	// decorations this backend is allowed to emit. Empty means the backend gets
	// no allowance and every non-canonical response field is rejected.
	vendorExtensions string
}

// resolveOpenAIBackendDialect captures request-shaping differences between
// OpenAI-compatible backends. A nil profile is the legacy local-vLLM path, where
// reasoning_effort must stay in chat_template_kwargs.
func resolveOpenAIBackendDialect(profile *config.ProviderProfile) openAIBackendDialect {
	if profile == nil {
		return newOpenAIBackendDialect(openAIBackendDialectVLLM)
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
		// Azure OpenAI and Azure AI Foundry speak the OpenAI request contract
		// but decorate every response with their own fields.
		return newOpenAIBackendDialect(openAIBackendDialectAzureOpenAI)
	}
	return newOpenAIBackendDialect(openAIBackendDialectGenericOpenAICompat)
}

func isAzureOpenAIHost(host string) bool {
	if host == "" {
		return false
	}
	for _, suffix := range azureOpenAIHostSuffixes {
		if strings.HasSuffix(host, suffix) {
			return true
		}
	}
	return false
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
		// Request shaping is deliberately identical to the generic
		// OpenAI-compatible dialect; only response decoration differs, which is
		// all issue #3496 covers. Any Azure request-side quirk needs its own
		// evidence before it is claimed here.
		dialect.vendorExtensions = llmprotocol.VendorAzure
	}
	return dialect
}

// vendorExtensionProvider reports which provider's documented response
// decorations the response decoder may ignore for this backend.
func (d openAIBackendDialect) vendorExtensionProvider() string {
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
