//go:build !dev

package extproc

func extractRequestAutoStore(ctx *RequestContext) (bool, bool) {
	return false, false
}
