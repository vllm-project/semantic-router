package extproc

import (
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type openAIBackendDialectKind string

const (
	openAIBackendDialectOfficialOpenAI      openAIBackendDialectKind = "official_openai"
	openAIBackendDialectOfficialDeepSeek    openAIBackendDialectKind = "official_deepseek"
	openAIBackendDialectVLLM                openAIBackendDialectKind = "vllm"
	openAIBackendDialectGenericOpenAICompat openAIBackendDialectKind = "generic_openai_compatible"
)

type openAIBackendDialect struct {
	kind                            openAIBackendDialectKind
	supportsTopLevelReasoningEffort bool
	supportsTopLevelDeepSeekThink   bool
}

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
		return newOpenAIBackendDialect(openAIBackendDialectOfficialDeepSeek)
	default:
		return newOpenAIBackendDialect(openAIBackendDialectGenericOpenAICompat)
	}
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
	}
	return dialect
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
