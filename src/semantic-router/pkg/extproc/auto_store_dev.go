//go:build dev

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func extractRequestAutoStore(ctx *RequestContext) (bool, bool) {
	if ctx == nil || ctx.ResponseAPICtx == nil || !ctx.ResponseAPICtx.IsResponseAPIRequest || ctx.ResponseAPICtx.OriginalRequest == nil || ctx.ResponseAPICtx.OriginalRequest.AutoStore == nil {
		return false, false
	}

	logging.Infof(
		"extractAutoStore: Using Response API request auto_store=%v (request_id=%s)",
		*ctx.ResponseAPICtx.OriginalRequest.AutoStore,
		ctx.RequestID,
	)
	return *ctx.ResponseAPICtx.OriginalRequest.AutoStore, true
}
