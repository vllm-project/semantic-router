package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// cloudflareWorkersAIProviderType is the catalog provider id for Cloudflare
// Workers AI. The provider id is the only reliable selector for its failure
// envelope: the account-scoped base URL also fronts Workers AI's natively
// documented run surface, which does not share this errors[] array, so a host
// match would misidentify a profile pointed at the native surface.
const cloudflareWorkersAIProviderType = "cloudflare-workers-ai"

// snowflakeCortexProviderType is the catalog provider id for Snowflake Cortex
// AI. Its account-scoped host also fronts the Snowflake SQL API, whose bodies do
// not share this failure envelope, so the provider id is the only reliable
// selector.
const snowflakeCortexProviderType = "snowflake-cortex"

// openRouterProviderType is the catalog provider id for OpenRouter. Like the
// other failure-envelope vendors it is selected by provider id only: a profile
// that points an openai-type provider at OpenRouter's host keeps the canonical
// error object.
const openRouterProviderType = "openrouter"

// resolveResponseVendor reports which provider's own failure envelope the
// response decoder reads for this backend. It is deliberately independent of
// reasoning transport: the envelope is a property of who serves the response,
// not of how a request is shaped.
//
// Members a wire contract does not declare need no vendor: every backend drops
// and reports them. An empty vendor decodes the wire format's canonical error
// object.
func resolveResponseVendor(profile *config.ProviderProfile) llmprotocol.ResponseVendor {
	if profile == nil {
		return ""
	}
	switch strings.ToLower(strings.TrimSpace(profile.Type)) {
	case cloudflareWorkersAIProviderType:
		return llmprotocol.ResponseVendorCloudflare
	case snowflakeCortexProviderType:
		return llmprotocol.ResponseVendorSnowflake
	case openRouterProviderType:
		return llmprotocol.ResponseVendorOpenRouter
	}
	return ""
}
