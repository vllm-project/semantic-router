package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
)

// responseObjectOwner derives the retention owner only from trusted
// process-local request state. Managed access and access-disabled public
// routing are intentionally separate ownership modes.
func (r *OpenAIRouter) responseObjectOwner(ctx *RequestContext) (responseapi.ResponseOwner, bool) {
	if r == nil || ctx == nil {
		return responseapi.ResponseOwner{}, false
	}
	if r.managedInferenceAccessEnabled() {
		if ctx.InferenceAccess == nil {
			return responseapi.ResponseOwner{}, false
		}
		ctx.InferenceAccess.mu.Lock()
		tenant := ctx.InferenceAccess.tenant
		ctx.InferenceAccess.mu.Unlock()
		owner := responseapi.ResponseOwner{
			Mode:        responseapi.ResponseOwnerAuthenticated,
			NamespaceID: tenant.NamespaceID, APIKeyID: tenant.APIKeyID, UserID: tenant.UserID,
		}
		return owner, owner.Valid()
	}

	generation, ok := routingcontext.GenerationFrom(ctx.TraceContext)
	if !ok {
		return responseapi.ResponseOwner{}, false
	}
	owner := responseapi.ResponseOwner{
		Mode:        responseapi.ResponseOwnerAnonymousPublicNamespace,
		NamespaceID: generation.NamespaceID,
	}
	return owner, owner.Valid()
}
