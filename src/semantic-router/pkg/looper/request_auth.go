package looper

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

const (
	looperRequestTokenBytes     = 32
	looperRequestTokenHexLength = looperRequestTokenBytes * 2
)

// RequestAuthenticator proves that a request was created by a trusted router
// runtime's internal Looper client. Its token is runtime-owned and is never
// serialized into user configuration; deployments may share it across trusted
// replicas through a secret environment variable.
type RequestAuthenticator struct {
	token [looperRequestTokenBytes]byte
}

// NewRequestAuthenticator creates a fresh process-local 256-bit internal
// request token.
func NewRequestAuthenticator() (*RequestAuthenticator, error) {
	return newRequestAuthenticator(rand.Reader)
}

func newRequestAuthenticator(random io.Reader) (*RequestAuthenticator, error) {
	authenticator := &RequestAuthenticator{}
	if _, err := io.ReadFull(random, authenticator.token[:]); err != nil {
		return nil, fmt.Errorf("generate looper request token: %w", err)
	}
	return authenticator, nil
}

// NewRequestAuthenticatorFromSharedSecret creates an authenticator from a
// deployment-owned 256-bit secret encoded as exactly 64 hexadecimal
// characters. The supplied value is never included in returned errors.
func NewRequestAuthenticatorFromSharedSecret(secret string) (*RequestAuthenticator, error) {
	if len(secret) != looperRequestTokenHexLength {
		return nil, fmt.Errorf("looper shared secret must be exactly %d hexadecimal characters", looperRequestTokenHexLength)
	}

	authenticator := &RequestAuthenticator{}
	decoded, err := hex.Decode(authenticator.token[:], []byte(secret))
	if err != nil || decoded != looperRequestTokenBytes {
		return nil, fmt.Errorf("looper shared secret must be exactly %d hexadecimal characters", looperRequestTokenHexLength)
	}
	return authenticator, nil
}

// Apply marks an outbound request as an authenticated internal Looper request.
// Callers should invoke this after applying user-configured headers so those
// headers cannot override the runtime-owned credentials.
func (a *RequestAuthenticator) Apply(requestHeaders http.Header) {
	if a == nil || requestHeaders == nil {
		return
	}
	requestHeaders.Set(headers.VSRLooperRequest, "true")
	requestHeaders.Set(headers.VSRLooperSecret, hex.EncodeToString(a.token[:]))
}

// Authenticate validates an inbound internal-request marker and token.
func (a *RequestAuthenticator) Authenticate(marker, token string) bool {
	if a == nil || marker != "true" || len(token) != looperRequestTokenHexLength {
		return false
	}
	decoded := make([]byte, looperRequestTokenBytes)
	n, err := hex.Decode(decoded, []byte(token))
	if err != nil || n != looperRequestTokenBytes {
		return false
	}
	return subtle.ConstantTimeCompare(decoded, a.token[:]) == 1
}
