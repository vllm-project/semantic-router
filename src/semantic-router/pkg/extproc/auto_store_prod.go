//go:build !dev

package extproc

func extractRequestAutoStore(ctx *RequestContext) (bool, bool) {
	if ctx == nil || ctx.ResponseAPICtx == nil || !ctx.ResponseAPICtx.IsResponseAPIRequest || ctx.ResponseAPICtx.OriginalRequest == nil || ctx.ResponseAPICtx.OriginalRequest.AutoStore == nil {
		return false, false
	}

	return *ctx.ResponseAPICtx.OriginalRequest.AutoStore, true
}
