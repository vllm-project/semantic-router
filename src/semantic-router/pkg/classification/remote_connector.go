package classification

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

// remoteConnector connects to a remote deployment through its connection
// block; timeout and retries apply when the deployment sets none.
func remoteConnector(deployment config.ModelDeployment, timeout time.Duration, retries int, maxErrorBytes int64) (*connector.Client, error) {
	return connector.ForDeployment(deployment, connector.Options{
		AttemptTimeout:   timeout,
		MaxRetries:       retries,
		MaxRequestBytes:  config.DefaultClassifyMaxRequestBytes,
		MaxResponseBytes: config.DefaultClassifyMaxResponseBytes,
		MaxErrorBytes:    maxErrorBytes,
	})
}

// externalClassifyDeployment is a deprecated external[] model asked through
// the classify API; a deadline replaces its llm_timeout_seconds.
func externalClassifyDeployment(external *config.ExternalModelConfig, deadline time.Duration) config.ModelDeployment {
	if external == nil {
		return config.ModelDeployment{}
	}
	deployment := config.ExternalModelDeployment(*external, config.ClassifyProvider)
	if deadline > 0 {
		deployment.TimeoutMs = int(deadline / time.Millisecond)
	}
	return deployment
}

// externalChatDeployment is a deprecated external[] model asked through the
// chat completions API.
func externalChatDeployment(external *config.ExternalModelConfig) config.ModelDeployment {
	if external == nil {
		return config.ModelDeployment{}
	}
	return config.ExternalModelDeployment(*external, config.OpenAIChatProvider)
}
