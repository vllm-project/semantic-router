package extproc

import (
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// azureOpenAIProviderType is the catalog provider id for Azure-hosted OpenAI
// deployments. A profile declaring it is the canonical Azure configuration.
const azureOpenAIProviderType = "azure-openai"

// cloudflareWorkersAIProviderType is the catalog provider id for Cloudflare
// Workors AI. The provider id is the only reliable selector for its response
// contract: the account-scoped base URL also fronts Workors AI's natively
// documented run surface, which does not share these decorations or this
// errors[] array, so a host match would misidentify a profile pointed at the
// native surface.
const cloudflareWorkersAIProviderType = "cloudflare-workers-ai"

// snowflakeCortexProviderType is the catalog provider id for Snowflake Cortex
// AI. Its account-scoped host also fronts the Snowflake SQL API, whose bodies do
// not share this failure envelope, so the provider id is the only reliable
// selector.
const snowflakeCortexProviderType = "snowflake-cortex"

// openRouterProviderType is the catalog provider id for OpenRouter. Like the
// other failure-envelope vendors it is selected by provider id only: a profile
// that points an openai-type provider at OpenRouter's host keeps the strict
// canonical contract.
const openRouterProviderType = "openrouter"

// anthropicMessagesProtocol is the catalog protocol of Anthropic-format
// backends, whose relays need the Anthropic gateway contract.
const anthropicMessagesProtocol = "anthropic/messages@1"

// azureOpenAIHostSuffixes cover a profile pointed at an Azure endpoint without
// declaring the Azure provider type. Azure resources are per-tenant subdomains,
// so these are matched by suffix rather than exact host.
var azureOpenAIHostSuffixes = []string{
	".openai.azure.com",
	".services.ai.azure.com",
	".cognitiveservices.azure.com",
}

// resolveResponseVendor reports which provider's documented response
// decorations the response decoder may ignore for this backend. It is
// deliberately independent of reasoning transport: response decoration is a
// property of who serves the response, not of how a request is shaped.
//
// The selected vendor also governs the transport-error envelope, because a
// provider that decorates accepted responses can report failures in a shape the
// canonical OpenAI error wire cannot carry.
//
// An empty vendor is the strict default, so every backend that is not
// positively identified keeps the canonical response contract.
func resolveResponseVendor(profile *config.ProviderProfile) llmprotocol.ResponseVendor {
	if profile == nil {
		return ""
	}
	if strings.EqualFold(strings.TrimSpace(profile.Type), azureOpenAIProviderType) {
		return llmprotocol.ResponseVendorAzure
	}
	if strings.EqualFold(strings.TrimSpace(profile.Type), cloudflareWorkersAIProviderType) {
		return llmprotocol.ResponseVendorCloudflare
	}
	if strings.EqualFold(strings.TrimSpace(profile.Type), snowflakeCortexProviderType) {
		return llmprotocol.ResponseVendorSnowflake
	}
	if strings.EqualFold(strings.TrimSpace(profile.Type), openRouterProviderType) {
		return llmprotocol.ResponseVendorOpenRouter
	}
	if vendor := compatGatewayResponseVendor(profile); vendor != "" {
		return vendor
	}
	if isAzureOpenAIHost(normalizedProfileHost(profile)) {
		return llmprotocol.ResponseVendorAzure
	}
	return ""
}

// compatGatewayResponseVendor selects the relay contract for the backend's wire
// format. A backend tolerates a gateway's own response fields only when the
// operator opts in, so every other backend keeps the strict contract even if
// it ever starts emitting such fields.
func compatGatewayResponseVendor(profile *config.ProviderProfile) llmprotocol.ResponseVendor {
	if !profile.CompatGateway {
		return ""
	}
	switch profile.Protocol {
	case anthropicMessagesProtocol:
		return llmprotocol.ResponseVendorAnthropicCompatGateway
	case "", "openai/chat-completions@1", "openai/responses@1":
		return llmprotocol.ResponseVendorOpenAICompatGateway
	default:
		return ""
	}
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

func normalizedProfileHost(profile *config.ProviderProfile) string {
	if profile == nil || profile.BaseURL == "" {
		return ""
	}
	parsed, err := url.Parse(profile.BaseURL)
	if err != nil {
		return ""
	}
	return strings.ToLower(parsed.Hostname())
}
