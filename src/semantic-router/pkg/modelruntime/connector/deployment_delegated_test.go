package connector

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
)

func TestDeploymentAuthorizerPresentsOnlyTheRequestsDelegatedCredential(t *testing.T) {
	prefix := ""
	authorize := DeploymentAuthorizer(&config.DeploymentAuth{Credential: config.CredentialDelegated, Header: "X-Gateway-Key", Prefix: &prefix}, nil)
	request, _ := http.NewRequest(http.MethodPost, "https://decisions.example/v1/systemone", nil)
	if err := authorize(principal.ForRequest(context.Background(), "alice-key"), request); err != nil {
		t.Fatal(err)
	}
	if got := request.Header.Get("X-Gateway-Key"); got != "alice-key" {
		t.Fatalf("X-Gateway-Key = %q, want the request's credential in the deployment's header and prefix", got)
	}
	for _, test := range []struct {
		name string
		ctx  context.Context
		want error
	}{
		{"a request without a credential", principal.ForRequest(context.Background(), ""), principal.ErrCredentialMissing},
		{"a Router-own call", context.Background(), principal.ErrRouterCall},
	} {
		refused, _ := http.NewRequest(http.MethodPost, "https://decisions.example/v1/systemone", nil)
		if err := authorize(test.ctx, refused); !errors.Is(err, test.want) {
			t.Fatalf("%s: error = %v, want %v", test.name, err, test.want)
		}
		if refused.Header.Get("X-Gateway-Key") != "" {
			t.Fatalf("%s: a refused call carried a credential", test.name)
		}
	}
}
