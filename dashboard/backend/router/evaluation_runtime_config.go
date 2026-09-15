package router

import (
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

func evaluationServiceEndpoint(
	configured config.EvaluationServiceEndpointConfig,
) *evaluationplane.ServiceEndpoint {
	if !configured.Configured() {
		return nil
	}
	return &evaluationplane.ServiceEndpoint{
		SchemaVersion:  evaluationplane.SchemaVersion,
		URL:            configured.URL,
		APIKey:         &evaluationplane.SecretRef{SchemaVersion: evaluationplane.SchemaVersion, Env: configured.APIKeyEnv},
		TimeoutSeconds: configured.Timeout.Seconds(),
	}
}
