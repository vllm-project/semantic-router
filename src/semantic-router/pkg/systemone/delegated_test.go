package systemone

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
)

func TestForClientsKeepsTheCredentialOnlyBesideTheAuthenticatedUser(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Authz.DelegatedCredential.Enabled = true
	for _, test := range []struct {
		name, user, want string
	}{
		{"an authenticated user", "alice", "alice-key"},
		{"no authenticated user", "", ""},
	} {
		var credential, header string
		var forUser bool
		handler := ForClients(cfg, func(_ http.ResponseWriter, r *http.Request) {
			credential, forUser = principal.RequestCredential(r.Context())
			header = r.Header.Get("x-vsr-delegated-credential")
		})
		request := httptest.NewRequest(http.MethodPost, "/v1/systemone", nil)
		request.Header.Set("x-vsr-delegated-credential", "alice-key")
		if test.user != "" {
			request.Header.Set("x-authz-user-id", test.user)
		}
		handler(httptest.NewRecorder(), request)
		if credential != test.want || !forUser || header != "" {
			t.Fatalf("%s: credential %q (for a user: %v), header left %q; want %q", test.name, credential, forUser, header, test.want)
		}
	}
}
