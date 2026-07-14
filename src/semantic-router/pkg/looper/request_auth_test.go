package looper

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

type failingEntropyReader struct{}

func (failingEntropyReader) Read([]byte) (int, error) {
	return 0, errors.New("entropy unavailable")
}

func TestNewRequestAuthenticatorFailsClosedWithoutEntropy(t *testing.T) {
	if _, err := newRequestAuthenticator(failingEntropyReader{}); err == nil {
		t.Fatal("newRequestAuthenticator() error = nil")
	}
}

func TestRequestAuthenticatorAcceptsOnlyItsOwnToken(t *testing.T) {
	authenticator, err := NewRequestAuthenticator()
	if err != nil {
		t.Fatalf("NewRequestAuthenticator() error = %v", err)
	}

	requestHeaders := make(http.Header)
	authenticator.Apply(requestHeaders)
	marker := requestHeaders.Get(headers.VSRLooperRequest)
	token := requestHeaders.Get(headers.VSRLooperSecret)
	if token == "" {
		t.Fatal("Apply() did not set an authentication token")
	}
	if !authenticator.Authenticate(marker, token) {
		t.Fatal("Authenticate() rejected the authenticator's own token")
	}

	tests := []struct {
		name   string
		marker string
		token  string
	}{
		{name: "missing marker", token: token},
		{name: "wrong marker", marker: "false", token: token},
		{name: "missing token", marker: "true"},
		{name: "wrong token", marker: "true", token: token + "x"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if authenticator.Authenticate(test.marker, test.token) {
				t.Fatal("Authenticate() accepted invalid credentials")
			}
		})
	}
}

func TestRequestAuthenticatorsUseDistinctTokens(t *testing.T) {
	first, err := NewRequestAuthenticator()
	if err != nil {
		t.Fatalf("NewRequestAuthenticator() first error = %v", err)
	}
	second, err := NewRequestAuthenticator()
	if err != nil {
		t.Fatalf("NewRequestAuthenticator() second error = %v", err)
	}

	requestHeaders := make(http.Header)
	first.Apply(requestHeaders)
	if second.Authenticate(
		requestHeaders.Get(headers.VSRLooperRequest),
		requestHeaders.Get(headers.VSRLooperSecret),
	) {
		t.Fatal("a different authenticator accepted the first authenticator's token")
	}
}

func TestRequestAuthenticatorsShareDeploymentSecret(t *testing.T) {
	secret := strings.Repeat("0123456789abcdef", 4)
	first, err := NewRequestAuthenticatorFromSharedSecret(secret)
	if err != nil {
		t.Fatalf("NewRequestAuthenticatorFromSharedSecret() first error = %v", err)
	}
	second, err := NewRequestAuthenticatorFromSharedSecret(secret)
	if err != nil {
		t.Fatalf("NewRequestAuthenticatorFromSharedSecret() second error = %v", err)
	}

	requestHeaders := make(http.Header)
	first.Apply(requestHeaders)
	if !second.Authenticate(
		requestHeaders.Get(headers.VSRLooperRequest),
		requestHeaders.Get(headers.VSRLooperSecret),
	) {
		t.Fatal("an authenticator initialized from the same shared secret rejected the token")
	}
}

func TestNewRequestAuthenticatorFromSharedSecretRejectsInvalidValues(t *testing.T) {
	tests := []struct {
		name   string
		secret string
	}{
		{name: "empty"},
		{name: "too short", secret: strings.Repeat("a", looperRequestTokenHexLength-1)},
		{name: "non hexadecimal", secret: strings.Repeat("g", looperRequestTokenHexLength)},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewRequestAuthenticatorFromSharedSecret(test.secret)
			if err == nil {
				t.Fatal("NewRequestAuthenticatorFromSharedSecret() error = nil")
			}
			if test.secret != "" && strings.Contains(err.Error(), test.secret) {
				t.Fatal("validation error exposed the supplied shared secret")
			}
		})
	}
}
