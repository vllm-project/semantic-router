package extproc

import (
	"net/url"
	"strings"

	"github.com/tidwall/sjson"
)

// Azure OpenAI clients select a model by deployment, as in
// POST /openai/deployments/{deployment}/chat/completions?api-version=..., and
// send their key in an api-key header.
const (
	azureDeploymentsPath = "/openai/deployments"
	azureChatOperation   = "/chat/completions"
	azureAPIKeyHeader    = "api-key"
	azureResponsesPath   = "/openai/responses"
	azureV1ResponsesPath = "/openai/v1/responses"
	azureV1ChatPath      = "/openai/v1/chat/completions"
)

func isAzureResponsesCollection(path string) bool {
	switch normalizeRequestPath(path) {
	case azureResponsesPath, azureV1ResponsesPath:
		return true
	default:
		return false
	}
}

func isAzureOpenAIPath(path string) bool {
	normalized := normalizeRequestPath(path)
	return normalized == "/openai" || strings.HasPrefix(normalized, "/openai/")
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
