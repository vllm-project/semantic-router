package connector

import (
	"context"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
)

func TestDeploymentAuthorizerForwardsTheRequestsHeadersOverStaticOnes(t *testing.T) {
	t.Setenv("FORWARD_TEST_KEY", "router-key")
	auth := &config.DeploymentAuth{Header: "X-Gateway-Key", APIKeyEnv: "FORWARD_TEST_KEY"}
	authorize := DeploymentAuthorizer(auth, map[string]string{"X-User-Id": "router", "X-Team": "search"}, []string{"x-user-id", "X-Session-Id"})
	call := func(ctx context.Context) http.Header {
		t.Helper()
		request, _ := http.NewRequest(http.MethodPost, "https://decisions.example/v1/systemone", nil)
		if err := authorize(ctx, request); err != nil {
			t.Fatal(err)
		}
		return request.Header
	}

	alice := call(principal.ForRequestWithHeaders(context.Background(), "", map[string]string{"x-user-id": "alice", "x-gateway-key": "client-key"}))
	if alice.Get("X-User-Id") != "alice" || len(alice.Values("X-User-Id")) != 1 {
		t.Fatalf("X-User-Id = %v, want the client's value over the static one", alice.Values("X-User-Id"))
	}
	if alice.Get("X-Team") != "search" || alice.Get("X-Gateway-Key") != "Bearer router-key" {
		t.Fatalf("static headers and the credential stay the deployment's: %v", alice)
	}
	if _, sent := alice["X-Session-Id"]; sent {
		t.Fatalf("a header the client did not send is not sent: %v", alice)
	}

	for name, ctx := range map[string]context.Context{
		"a request without the header": principal.ForRequestWithHeaders(context.Background(), "", nil),
		"a Router-own call":            context.Background(),
	} {
		if header := call(ctx); header.Get("X-User-Id") != "router" || header.Get("X-Gateway-Key") != "Bearer router-key" {
			t.Fatalf("%s keeps the static value and the credential: %v", name, header)
		}
	}
}
