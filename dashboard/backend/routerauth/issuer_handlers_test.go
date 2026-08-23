package routerauth

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/golang-jwt/jwt/v5"
)

type issuerTestSigner struct{}

func (issuerTestSigner) KeyID() string { return "dashboard-key-1" }
func (issuerTestSigner) PublicJWK() PublicJWK {
	return PublicJWK{KeyType: "OKP", Curve: "Ed25519", Use: "sig", Algorithm: "EdDSA", KeyID: "dashboard-key-1", X: "public-key"}
}
func (issuerTestSigner) Sign(jwt.Claims) (string, error) { return "assertion", nil }

func TestRegisterIssuerDiscoveryPublishesOIDCMetadataAndJWKS(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	RegisterIssuerDiscovery(mux, "https://dashboard.example.test", issuerTestSigner{})

	discovery := httptest.NewRecorder()
	mux.ServeHTTP(discovery, httptest.NewRequest(http.MethodGet, "/.well-known/openid-configuration", nil))
	if discovery.Code != http.StatusOK || discovery.Header().Get("Cache-Control") != "public, max-age=300" {
		t.Fatalf("discovery status=%d headers=%v", discovery.Code, discovery.Header())
	}
	var metadata map[string]any
	if err := json.Unmarshal(discovery.Body.Bytes(), &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata["issuer"] != "https://dashboard.example.test" ||
		metadata["jwks_uri"] != "https://dashboard.example.test/.well-known/jwks.json" {
		t.Fatalf("discovery metadata = %#v", metadata)
	}

	keys := httptest.NewRecorder()
	mux.ServeHTTP(keys, httptest.NewRequest(http.MethodGet, "/.well-known/jwks.json", nil))
	if keys.Code != http.StatusOK || !json.Valid(keys.Body.Bytes()) {
		t.Fatalf("JWKS status=%d body=%s", keys.Code, keys.Body.String())
	}
}

func TestRegisterIssuerDiscoveryRejectsMutation(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	RegisterIssuerDiscovery(mux, "https://dashboard.example.test", issuerTestSigner{})
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/.well-known/jwks.json", nil))
	if recorder.Code != http.StatusMethodNotAllowed || recorder.Header().Get("Allow") != "GET, HEAD" {
		t.Fatalf("status=%d allow=%q", recorder.Code, recorder.Header().Get("Allow"))
	}
}
