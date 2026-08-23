package issuerverifier

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

type roundTripperStub struct {
	mu        sync.Mutex
	responses map[string]string
	requests  []string
}

func (transport *roundTripperStub) RoundTrip(request *http.Request) (*http.Response, error) {
	transport.mu.Lock()
	defer transport.mu.Unlock()
	target := request.URL.String()
	transport.requests = append(transport.requests, target)
	body, found := transport.responses[target]
	if !found {
		return nil, fmt.Errorf("unexpected request %s", target)
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    request,
	}, nil
}

func TestRemoteKeySourceDiscoversCachesAndInvalidatesKeys(t *testing.T) {
	public, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	discoveryURL := "https://issuer.example/.well-known/openid-configuration"
	jwksURL := "https://issuer.example/.well-known/jwks.json"
	transport := &roundTripperStub{responses: map[string]string{
		discoveryURL: `{"issuer":"https://issuer.example","jwks_uri":"` + jwksURL +
			`","id_token_signing_alg_values_supported":["EdDSA"]}`,
		jwksURL: `{"keys":[{"kty":"OKP","crv":"Ed25519","kid":"signing-1",` +
			`"use":"sig","key_ops":["verify"],"alg":"EdDSA","x":"` +
			base64.RawURLEncoding.EncodeToString(public) + `"}]}`,
	}}
	now := time.Date(2026, 8, 23, 1, 2, 3, 0, time.UTC)
	source, err := NewRemoteKeySource(RemoteKeySourceOptions{
		Transport: transport, CacheTTL: 5 * time.Minute, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = source.Close() })
	issuer := TrustedIssuer{
		ID: testIssuerID, Issuer: "https://issuer.example", Kind: IssuerOIDC,
		DiscoveryURL: discoveryURL, Audiences: []string{ManagementAudience},
		ClaimMapping: map[string]string{}, AssuranceMapping: map[string]string{}, Revision: 1,
	}
	for attempt := 0; attempt < 2; attempt++ {
		set, err := source.Keys(context.Background(), issuer)
		if err != nil || len(set.Keys) != 1 || set.Keys["signing-1"].Algorithm != "EdDSA" {
			t.Fatalf("Keys() = %+v, %v", set, err)
		}
	}
	if len(transport.requests) != 2 {
		t.Fatalf("cached requests = %v", transport.requests)
	}
	source.Invalidate(testIssuerID)
	if _, err := source.Keys(context.Background(), issuer); err != nil {
		t.Fatal(err)
	}
	if len(transport.requests) != 4 {
		t.Fatalf("invalidated requests = %v", transport.requests)
	}
	if err := source.Refresh(context.Background(), issuer); err != nil {
		t.Fatal(err)
	}
	if len(transport.requests) != 6 {
		t.Fatalf("refreshed requests = %v", transport.requests)
	}
}

func TestRemoteKeySourceRejectsDiscoveryIssuerMismatch(t *testing.T) {
	discoveryURL := "https://issuer.example/.well-known/openid-configuration"
	transport := &roundTripperStub{responses: map[string]string{
		discoveryURL: `{"issuer":"https://attacker.example","jwks_uri":"https://issuer.example/jwks",` +
			`"id_token_signing_alg_values_supported":["EdDSA"]}`,
	}}
	source, err := NewRemoteKeySource(RemoteKeySourceOptions{Transport: transport})
	if err != nil {
		t.Fatal(err)
	}
	issuer := TrustedIssuer{
		ID: testIssuerID, Issuer: "https://issuer.example", Kind: IssuerOIDC,
		DiscoveryURL: discoveryURL, Audiences: []string{ManagementAudience},
		ClaimMapping: map[string]string{}, AssuranceMapping: map[string]string{}, Revision: 1,
	}
	if _, err := source.Keys(context.Background(), issuer); err == nil {
		t.Fatal("Keys() accepted mismatched discovery issuer")
	}
}
