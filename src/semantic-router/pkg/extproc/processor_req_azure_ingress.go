package extproc

import (
	"net/url"
	"strings"

	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Azure OpenAI clients select a model by deployment, as in
// POST /openai/deployments/{deployment}/chat/completions?api-version=..., and
// send their key in an api-key header.
const (
	azureOpenAIPath      = "/openai"
	azureDeploymentsPath = "/openai/deployments"
	azureChatOperation   = "/chat/completions"
	azureAPIKeyHeader    = "api-key"
)

// azureBodyModelPaths are the Azure OpenAI routes whose request body names the
// model: the versionless v1 API, and Responses with an api-version query.
var azureBodyModelPaths = map[string]llmprotocol.WireFormat{
	"/openai/v1/chat/completions": llmprotocol.OpenAIChatV1,
	"/openai/v1/responses":        llmprotocol.OpenAIResponsesV1,
	"/openai/responses":           llmprotocol.OpenAIResponsesV1,
}

// azureIngressFormat returns the public wire format of a supported Azure
// OpenAI path, or "" for any other path.
func azureIngressFormat(path string) llmprotocol.WireFormat {
	normalizedPath := normalizeRequestPath(path)
	if format, ok := azureBodyModelPaths[normalizedPath]; ok {
		return format
	}
	if _, ok := azureChatDeployment(normalizedPath); ok {
		return llmprotocol.OpenAIChatV1
	}
	return ""
}

// azureChatDeployment returns the deployment named by an Azure Chat
// Completions path. Router model names such as vllm-sr/auto contain slashes,
// so the deployment is everything between the prefix and the operation.
func azureChatDeployment(path string) (string, bool) {
	rest, ok := strings.CutPrefix(normalizeRequestPath(path), azureDeploymentsPath+"/")
	if !ok {
		return "", false
	}
	escaped, ok := strings.CutSuffix(rest, azureChatOperation)
	if !ok {
		return "", false
	}
	deployment, err := url.PathUnescape(escaped)
	if err != nil || strings.TrimSpace(deployment) == "" {
		return "", false
	}
	return deployment, true
}

func isAzureOpenAIPath(normalizedPath string) bool {
	return normalizedPath == azureOpenAIPath ||
		strings.HasPrefix(normalizedPath, azureOpenAIPath+"/")
}

// withAzureDeploymentModel makes the deployment the request model. Azure's
// request body has no model field; the deployment in the path selects it.
func withAzureDeploymentModel(body []byte, path string) []byte {
	deployment, ok := azureChatDeployment(path)
	if !ok {
		return body
	}
	rewritten, err := sjson.SetBytes(body, "model", deployment)
	if err != nil {
		return body
	}
	return rewritten
}
