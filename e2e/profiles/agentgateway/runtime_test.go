package agentgateway

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
)

func TestFailSetupCollectsRouterDiagnostics(t *testing.T) {
	called := false
	opts := &framework.SetupOptions{
		CollectRouterDiagnostics: func(_ context.Context, namespace string) error {
			called = true
			if namespace != agentGatewayNamespace {
				t.Fatalf("diagnostics namespace = %q", namespace)
			}
			return nil
		},
	}

	wantErr := errors.New("setup failed")
	gotErr := (&Profile{}).failSetup(context.Background(), opts, &setupState{}, wantErr)
	if !errors.Is(gotErr, wantErr) {
		t.Fatalf("failSetup() error = %v", gotErr)
	}
	if !called {
		t.Fatal("failSetup() did not collect diagnostics before cleanup")
	}
}
