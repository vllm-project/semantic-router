package config

import "fmt"

// providerTypesWithResponseContract are selected for their own response
// contract by provider type, ahead of any relay opt-in.
var providerTypesWithResponseContract = map[string]bool{
	"azure-openai":          true,
	"cloudflare-workers-ai": true,
	"openrouter":            true,
	"snowflake-cortex":      true,
}

// ValidateCompatGateway reports why compat_gateway cannot take effect on a
// backend of the given provider type and wire protocol, or nil when it does.
func ValidateCompatGateway(providerType, protocol string) error {
	if providerTypesWithResponseContract[providerType] {
		return fmt.Errorf("compat_gateway has no effect on %s backends, which keep their own response contract", providerType)
	}
	switch protocol {
	case "openai/chat-completions@1", "openai/responses@1", "anthropic/messages@1":
		return nil
	}
	return fmt.Errorf("compat_gateway applies only to OpenAI and Anthropic wire formats, not %s", protocol)
}
