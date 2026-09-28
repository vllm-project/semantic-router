package internalauth

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
)

const tokenBytes = 32

var (
	processToken       = generateToken()
	processTokenDigest = sha256.Sum256([]byte(processToken))
)

// Token returns the process-local credential used to authenticate internal
// request context. Callers must never persist or log the returned value.
func Token() string {
	return processToken
}

// Authenticate reports whether candidate is the current process-local token.
// Comparing fixed-size digests avoids leaking the token length through the
// comparison path.
func Authenticate(candidate string) bool {
	candidateDigest := sha256.Sum256([]byte(candidate))
	return subtle.ConstantTimeCompare(processTokenDigest[:], candidateDigest[:]) == 1
}

func generateToken() string {
	value := make([]byte, tokenBytes)
	if _, err := rand.Read(value); err != nil {
		panic("failed to generate internal request token")
	}
	return base64.RawURLEncoding.EncodeToString(value)
}
