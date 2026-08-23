package routerauth

import (
	"encoding/json"
	"net/http"
)

// RegisterIssuerDiscovery exposes only public verification metadata. It never
// serves private material or accepts identity mutations.
func RegisterIssuerDiscovery(mux *http.ServeMux, issuerURL string, signer AssertionSigner) {
	if mux == nil || issuerURL == "" || signer == nil {
		return
	}
	write := func(payload any) http.HandlerFunc {
		return func(response http.ResponseWriter, request *http.Request) {
			if request.Method != http.MethodGet && request.Method != http.MethodHead {
				response.Header().Set("Allow", "GET, HEAD")
				http.Error(response, "Method not allowed", http.StatusMethodNotAllowed)
				return
			}
			response.Header().Set("Content-Type", "application/json")
			response.Header().Set("Cache-Control", "public, max-age=300")
			if request.Method == http.MethodHead {
				response.WriteHeader(http.StatusOK)
				return
			}
			_ = json.NewEncoder(response).Encode(payload)
		}
	}
	mux.HandleFunc("/.well-known/openid-configuration", write(map[string]any{
		"issuer":                                issuerURL,
		"jwks_uri":                              issuerURL + "/.well-known/jwks.json",
		"id_token_signing_alg_values_supported": []string{dashboardAssertionAlgorithm},
	}))
	mux.HandleFunc("/.well-known/jwks.json", write(map[string]any{"keys": []PublicJWK{signer.PublicJWK()}}))
}
