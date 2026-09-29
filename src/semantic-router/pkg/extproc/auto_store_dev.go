//go:build dev

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func extractRequestAutoStore(ctx *RequestContext) (bool, bool) {
	if ctx == nil || ctx.RequestAutoStore == nil {
		return false, false
	}

	logging.Infof(
		"extractAutoStore: Using Response API request auto_store=%v (request_id=%s)",
		*ctx.RequestAutoStore,
		ctx.RequestID,
	)
	return *ctx.RequestAutoStore, true
}
