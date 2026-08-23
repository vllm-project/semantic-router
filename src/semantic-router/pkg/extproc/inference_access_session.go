package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
)

type inferenceAuthenticationContext struct {
	session accessruntime.Session
	tenant  accessruntime.TenantContext
	source  accessruntime.AuthenticationSource
}

type inferenceAuthenticationContextKey struct{}

func withInferenceAuthentication(
	ctx context.Context,
	authentication accessruntime.Authentication,
) context.Context {
	return context.WithValue(ctx, inferenceAuthenticationContextKey{}, inferenceAuthenticationContext{
		session: authentication.Session,
		tenant:  authentication.Tenant,
		source:  authentication.Source,
	})
}

func inferenceAuthenticationFromContext(ctx context.Context) (inferenceAuthenticationContext, bool) {
	if ctx == nil {
		return inferenceAuthenticationContext{}, false
	}
	value, ok := ctx.Value(inferenceAuthenticationContextKey{}).(inferenceAuthenticationContext)
	if !ok || value.tenant.NamespaceID == "" || value.tenant.APIKeyID == "" {
		return inferenceAuthenticationContext{}, false
	}
	return value, true
}
