//go:build !dev

package extproc

func extractRequestAutoStore(ctx *RequestContext) (bool, bool) {
	if ctx == nil || ctx.SemanticRequest == nil || ctx.SemanticRequest.AutoStore == nil {
		return false, false
	}

	return *ctx.SemanticRequest.AutoStore, true
}
