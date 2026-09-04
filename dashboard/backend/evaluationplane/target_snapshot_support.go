package evaluationplane

import (
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type externalSelectorConfigFingerprint struct {
	Model                 string  `json:"model"`
	BackendType           string  `json:"backend_type"`
	Provider              string  `json:"provider"`
	ModelRole             string  `json:"model_role"`
	ProviderModelIDDigest string  `json:"provider_model_id_digest"`
	EndpointName          string  `json:"endpoint_name,omitempty"`
	UseChatTemplate       bool    `json:"use_chat_template,omitempty"`
	PromptTemplateDigest  string  `json:"prompt_template_digest,omitempty"`
	TimeoutSeconds        int     `json:"timeout_seconds,omitempty"`
	ParserType            string  `json:"parser_type,omitempty"`
	Threshold             float32 `json:"threshold,omitempty"`
	MaxTokens             int     `json:"max_tokens,omitempty"`
	Temperature           float64 `json:"temperature,omitempty"`
	MaxRequestBytes       int64   `json:"max_request_bytes,omitempty"`
	MaxResponseBytes      int64   `json:"max_response_bytes,omitempty"`
}

type externalSelectorTopologyFingerprint struct {
	AddressDigest string `json:"address_digest"`
	Port          int    `json:"port"`
	Protocol      string `json:"protocol,omitempty"`
}

const externalSelectorDefaultTimeoutSeconds = 5

func externalSelectorSupportModel(
	external routerconfig.ExternalModelConfig,
	backendType string,
) (SupportModel, bool) {
	model := strings.TrimSpace(external.Name)
	providerModel := strings.TrimSpace(external.ModelName)
	backendType = strings.TrimSpace(backendType)
	switch backendType {
	case routerconfig.ClassifierSignalTypeLLM:
		if providerModel == "" {
			return SupportModel{}, false
		}
	case routerconfig.ClassifierSignalTypeSequenceClassifier:
		providerModel = model
	default:
		return SupportModel{}, false
	}
	address := strings.TrimSpace(external.ModelEndpoint.Address)
	if model == "" || address == "" ||
		external.ModelEndpoint.Port < 1 || external.ModelEndpoint.Port > 65535 {
		return SupportModel{}, false
	}
	providerDigest := digestString(providerModel)
	configFingerprint := externalSelectorConfig(external, backendType, model, providerDigest)
	return SupportModel{
		Model: model, ProviderModelIDDigest: providerDigest,
		ConfigDigest: digestJSON(configFingerprint),
		BackendTopologyDigest: digestJSON(externalSelectorTopologyFingerprint{
			AddressDigest: digestString(address), Port: external.ModelEndpoint.Port,
			Protocol: normalizedExternalSelectorProtocol(external.ModelEndpoint.Protocol),
		}),
	}, true
}

func externalSelectorConfig(
	external routerconfig.ExternalModelConfig,
	backendType, model, providerDigest string,
) externalSelectorConfigFingerprint {
	config := externalSelectorConfigFingerprint{
		Model: model, BackendType: backendType,
		ModelRole: strings.TrimSpace(external.ModelRole), ProviderModelIDDigest: providerDigest,
		EndpointName:    strings.TrimSpace(external.ModelEndpoint.Name),
		TimeoutSeconds:  externalSelectorTimeoutSeconds(external),
		MaxRequestBytes: external.GetMaxRequestBytes(), MaxResponseBytes: external.GetMaxResponseBytes(),
	}
	if backendType != routerconfig.ClassifierSignalTypeLLM {
		return config
	}
	config.Provider = strings.TrimSpace(external.Provider)
	config.UseChatTemplate = external.ModelEndpoint.UseChatTemplate
	if promptTemplate := strings.TrimSpace(external.ModelEndpoint.PromptTemplate); promptTemplate != "" {
		config.PromptTemplateDigest = digestString(promptTemplate)
	}
	config.ParserType = strings.TrimSpace(external.ParserType)
	config.Threshold = external.Threshold
	config.MaxTokens = external.MaxTokens
	config.Temperature = external.Temperature
	return config
}

func externalSelectorTimeoutSeconds(external routerconfig.ExternalModelConfig) int {
	if external.TimeoutSeconds > 0 {
		return external.TimeoutSeconds
	}
	return externalSelectorDefaultTimeoutSeconds
}

func normalizedExternalSelectorProtocol(protocol string) string {
	protocol = strings.ToLower(strings.TrimSpace(protocol))
	if protocol == "" {
		return "http"
	}
	return protocol
}
