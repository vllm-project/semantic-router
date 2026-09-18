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
// An empty vendor is the strict default, so every backend that is not
// positively identified keeps the canonical response contract.
func resolveResponseVendor(profile *config.ProviderProfile) llmprotocol.ResponseVendor {
	if profile == nil {
		return ""
	}
	if strings.EqualFold(strings.TrimSpace(profile.Type), azureOpenAIProviderType) {
		return llmprotocol.ResponseVendorAzure
	}
	if isAzureOpenAIHost(normalizedProfileHost(profile)) {
		return llmprotocol.ResponseVendorAzure
	}
	return ""
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
