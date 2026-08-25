package providercatalog

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// BuiltinIntegrations returns the product metadata shipped by the control
// plane. Applications can append their own Integration implementations before
// constructing a Registry; inference processes never load this metadata.
func BuiltinIntegrations() []Integration {
	specs := []builtinIntegrationSpec{
		withResponses(privateProvider("vllm", "vLLM", "vLLM endpoint", lobeIcon("vllm", true))),
		withResponses(privateProvider("sglang", "SGLang", "SGLang endpoint", urlIcon("https://raw.githubusercontent.com/sgl-project/sgl-docs/main/favicon.png"))),
		withResponses(privateProvider("amd-atom", "AMD ATOM", "AMD ATOM endpoint", assetIcon("/amd.png"))),
		withResponses(privateProvider("openai-compatible", "OpenAI compatible", "OpenAI-compatible endpoint", lobeIcon("openai", false))),
		fixedProvider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", lobeIcon("openrouter", false)),
		openAIProvider(),
		anthropicProvider(),
		anthropicCompatibleProvider(),
		fixedProvider("google-ai-studio", "Google AI Studio", "https://generativelanguage.googleapis.com/v1beta/openai", lobeIcon("gemini", true)),
		fixedProvider("deepseek", "DeepSeek", "https://api.deepseek.com", lobeIcon("deepseek", true)),
		fixedProvider("mistral", "Mistral", "https://api.mistral.ai/v1", lobeIcon("mistral", true)),
		fixedProvider("groq", "Groq", "https://api.groq.com/openai/v1", lobeIcon("groq", false)),
		fixedProvider("together-ai", "Together AI", "https://api.together.ai/v1", lobeIcon("together", true)),
		fixedProvider("fireworks-ai", "Fireworks AI", "https://api.fireworks.ai/inference/v1", lobeIcon("fireworks", true)),
		fixedProvider("cerebras", "Cerebras", "https://api.cerebras.ai/v1", lobeIcon("cerebras", true)),
		fixedProvider("xai", "xAI", "https://api.x.ai/v1", lobeIcon("xai", false)),
		fixedProvider("perplexity", "Perplexity", "https://api.perplexity.ai", lobeIcon("perplexity", true)),
		fixedProvider("cohere", "Cohere", "https://api.cohere.com/compatibility/v1", lobeIcon("cohere", true)),
		fixedProvider("deepinfra", "DeepInfra", "https://api.deepinfra.com/v1/openai", lobeIcon("deepinfra", true)),
		fixedProvider("hugging-face", "Hugging Face", "https://router.huggingface.co/v1", lobeIcon("huggingface", true)),
		fixedProvider("nvidia-nim", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", lobeIcon("nvidia", true)),
		fixedProvider("sambanova", "SambaNova", "https://api.sambanova.ai/v1", lobeIcon("sambanova", true)),
		fixedProvider("dashscope", "DashScope", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", lobeIcon("qwen", true)),
		fixedProvider("minimax", "MiniMax", "https://api.minimax.io/v1", lobeIcon("minimax", true)),
		fixedProvider("moonshot", "Moonshot AI", "https://api.moonshot.ai/v1", lobeIcon("moonshot", false)),
		fixedProvider("zai", "Z.ai", "https://api.z.ai/api/paas/v4", lobeIcon("zai", false)),
		fixedProvider("novita", "Novita AI", "https://api.novita.ai/v3/openai", lobeIcon("novita", true)),
		fixedProvider("nebius", "Nebius AI Studio", "https://api.studio.nebius.com/v1", lobeIcon("nebius", false)),
		fixedProvider("featherless", "Featherless AI", "https://api.featherless.ai/v1", lobeIcon("featherless", true)),
		fixedProvider("friendli", "FriendliAI", "https://api.friendli.ai/serverless/v1", lobeIcon("friendli", false)),
		fixedProvider("vercel-ai-gateway", "Vercel AI Gateway", "https://ai-gateway.vercel.sh/v1", lobeIcon("vercel", false)),
		fixedProvider("cometapi", "CometAPI", "https://api.cometapi.com/v1", lobeIcon("cometapi", true)),
		privateProvider("ollama", "Ollama", "Ollama endpoint", lobeIcon("ollama", false)),
		privateProvider("lm-studio", "LM Studio", "LM Studio endpoint", lobeIcon("lmstudio", false)),
		privateProvider("xinference", "Xinference", "Xinference endpoint", lobeIcon("xinference", true)),
		privateProvider("nvidia-riva", "NVIDIA Riva", "NVIDIA Riva endpoint", lobeIcon("nvidia", true)),
		privateProvider("triton", "NVIDIA Triton", "NVIDIA Triton endpoint", lobeIcon("nvidia", true)),
		fixedProvider("sakana", "Sakana AI", "https://api.sakana.ai/v1", urlIcon("https://console.sakana.ai/icon.svg")),
		privateProvider("docker-model-runner", "Docker Model Runner", "Docker Model Runner endpoint", urlIcon("https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png")),
		privateProvider("lemonade", "Lemonade", "Lemonade endpoint", assetIcon("/amd.png")),
	}
	result := make([]Integration, len(specs))
	for index := range specs {
		spec := specs[index]
		// #nosec G115 -- specs is a fixed, compile-time provider catalog with far fewer than uint32 entries.
		spec.Order = uint32(index + 1)
		result[index] = IntegrationFunc(spec.definition)
	}
	return result
}

type builtinIntegrationSpec struct {
	ID                   string
	Name                 string
	Order                uint32
	Origin               Origin
	WireFormat           llmprotocol.WireFormat
	Credential           Credential
	Discovery            *Discovery
	Path                 string
	Icon                 Icon
	InterfaceID          string
	InterfaceLabel       string
	AdditionalInterfaces []Interface
}

func openAIProvider() builtinIntegrationSpec {
	spec := fixedProvider("openai", "OpenAI", "https://api.openai.com/v1", lobeIcon("openai", false))
	return withResponses(spec)
}

func withResponses(spec builtinIntegrationSpec) builtinIntegrationSpec {
	spec.AdditionalInterfaces = []Interface{{
		ID: "responses", Label: "Responses API", WireFormat: llmprotocol.OpenAIResponsesV1,
		Compiler:     Compiler{AdapterID: StaticBackendCompilerID, Config: map[string]any{"path": "/responses"}},
		Capabilities: []string{"file_input", "image_input", "reasoning", "streaming", "text", "tools"},
	}}
	return spec
}

func lobeIcon(value string, color bool) Icon { return Icon{Source: "lobe", Value: value, Color: color} }
func urlIcon(value string) Icon              { return Icon{Source: "url", Value: value, Color: true} }
func assetIcon(value string) Icon            { return Icon{Source: "asset", Value: value, Color: true} }

func fixedProvider(id, name, origin string, icon Icon) builtinIntegrationSpec {
	return builtinIntegrationSpec{
		ID: id, Name: name, Icon: icon,
		Origin:      Origin{Mode: OriginFixed, DefaultURL: origin},
		WireFormat:  llmprotocol.OpenAIChatV1,
		Credential:  Credential{Mode: CredentialRequired, AdapterID: "bearer", Label: "API key"},
		Discovery:   &Discovery{AdapterID: "openai.models.v1", Path: "/models"},
		Path:        "/chat/completions",
		InterfaceID: "chat", InterfaceLabel: "Chat Completions",
	}
}

func privateProvider(id, name, label string, icon Icon) builtinIntegrationSpec {
	spec := fixedProvider(id, name, "", icon)
	spec.Origin = Origin{Mode: OriginUserSupplied, Label: label, Hint: "Enter the API base URL."}
	spec.Credential.Mode = CredentialOptional
	return spec
}

func anthropicProvider() builtinIntegrationSpec {
	return builtinIntegrationSpec{
		ID: "anthropic", Name: "Anthropic", Icon: lobeIcon("anthropic", false),
		Origin:     Origin{Mode: OriginFixed, DefaultURL: "https://api.anthropic.com"},
		WireFormat: llmprotocol.AnthropicMessagesV1,
		Credential: Credential{Mode: CredentialRequired, AdapterID: "x-api-key", Label: "API key"},
		Discovery: &Discovery{
			AdapterID: "anthropic.models.v1", Path: "/v1/models",
			Headers: map[string]string{"Anthropic-Version": "2023-06-01"},
		},
		Path:        "/v1/messages",
		InterfaceID: "messages", InterfaceLabel: "Messages API",
	}
}

func anthropicCompatibleProvider() builtinIntegrationSpec {
	return builtinIntegrationSpec{
		ID: "anthropic-compatible", Name: "Anthropic compatible", Icon: lobeIcon("anthropic", false),
		Origin:     Origin{Mode: OriginUserSupplied, Label: "Anthropic-compatible endpoint", Hint: "Enter the API base URL."},
		WireFormat: llmprotocol.AnthropicMessagesV1,
		Credential: Credential{Mode: CredentialOptional, AdapterID: "x-api-key", Label: "API key"},
		Discovery: &Discovery{
			AdapterID: "anthropic.models.v1", Path: "/v1/models",
			Headers: map[string]string{"Anthropic-Version": "2023-06-01"},
		},
		Path:        "/v1/messages",
		InterfaceID: "messages", InterfaceLabel: "Messages API",
	}
}

func (spec builtinIntegrationSpec) definition() Definition {
	interfaces := []Interface{{
		ID: spec.InterfaceID, Label: spec.InterfaceLabel, Default: true, WireFormat: spec.WireFormat,
		Compiler:     Compiler{AdapterID: StaticBackendCompilerID, Config: map[string]any{"path": spec.Path}},
		Capabilities: []string{"image_input", "streaming", "text", "tools"},
	}}
	interfaces = append(interfaces, spec.AdditionalInterfaces...)
	return Definition{
		ID: spec.ID, Order: spec.Order,
		Display: Display{
			Name: spec.Name, Description: "Connect " + spec.Name + " models.",
			Category: "Model APIs", Icon: spec.Icon,
		},
		Interfaces: interfaces,
		Credential: spec.Credential, Origin: spec.Origin, Discovery: spec.Discovery,
		Capabilities: []string{"image_input", "streaming", "text", "tools"},
	}
}
