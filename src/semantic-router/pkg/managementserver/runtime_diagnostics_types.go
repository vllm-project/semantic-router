package managementserver

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimediagnostics"
)

type RuntimeDiagnosticsService interface {
	Read(context.Context, string) (runtimediagnostics.Snapshot, error)
}

type RuntimeDiagnosticsRoutesOptions struct {
	Service       RuntimeDiagnosticsService
	Sessions      SessionAuthenticator
	Authorization Authorizer
	Now           func() time.Time
}
