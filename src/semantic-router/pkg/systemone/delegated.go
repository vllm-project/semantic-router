package systemone

import (
	"net/http"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
)

// The client-visible refusal of a System One request that a delegated
// upstream needs and that carries no delegated credential.
const (
	delegatedCredentialMissingCode    = "delegated_credential_missing" // #nosec G101 -- error code, not a credential
	delegatedCredentialMissingMessage = "Authentication failed. The selected upstream requires a delegated credential, which the request does not carry."
)

// ForClients makes the calls a public System One request causes ones made
// for that client, so a delegated upstream receives the credential the
// authenticator attached to it and a deployment that forwards request
// headers the client's values of them. The listener has dropped client
// copies of the authenticator's outputs unless it trusts identity headers,
// and the credential counts only beside the authenticated user. The header
// never travels further than this handler.
func ForClients(cfg *routerconfig.RouterConfig, next http.HandlerFunc) http.HandlerFunc {
	contract := cfg.Authz.DelegatedCredential
	userHeader := cfg.Authz.Identity.GetUserIDHeader()
	forwarded := cfg.DeploymentForwardedHeaders()
	return func(w http.ResponseWriter, r *http.Request) {
		credential := ""
		if contract.Enabled && strings.TrimSpace(r.Header.Get(userHeader)) != "" {
			credential = strings.TrimSpace(r.Header.Get(contract.GetHeader()))
		}
		r.Header.Del(contract.GetHeader())
		r.Header.Del(headers.VSRDelegatedCredential)
		next(w, r.WithContext(principal.ForRequestWithHeaders(r.Context(), credential, forwardedValues(r.Header, forwarded))))
	}
}

// forwardedValues are the request's values of the headers names lists, by
// lower-case name; nil when it carries none of them.
func forwardedValues(header http.Header, names []string) map[string]string {
	var values map[string]string
	for _, name := range names {
		if value := header.Get(name); value != "" {
			if values == nil {
				values = map[string]string{}
			}
			values[name] = value
		}
	}
	return values
}
