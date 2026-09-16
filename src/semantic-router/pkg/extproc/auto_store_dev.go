//go:build dev

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func extractRequestAutoStore(ctx *RequestContext) (bool, bool) {
	if ctx == nil || ctx.SemanticRequest == nil || ctx.SemanticRequest.AutoStore == nil {
		return false, false
	}

	logging.Infof(
		"extractAutoStore: Using Response API request auto_store=%v (request_id=%s)",
		*ctx.SemanticRequest.AutoStore,
		ctx.RequestID,
	)
	return *ctx.SemanticRequest.AutoStore, true
}
