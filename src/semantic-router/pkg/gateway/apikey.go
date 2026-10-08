package gateway

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// unauthorizedBody is the local Envoy template's 401 body for a missing or
// invalid client key.
const unauthorizedBody = `{"error":{"message":"Unauthorized: missing or invalid API key","type":"authentication_error","code":"invalid_api_key"}}`

// apiKeys checks client keys the way the template's Lua filter does: a Bearer
// token in authorization, or else api-key. A valid key authenticates the
// client to the Router only, so both headers are removed before routing.
type apiKeys struct {
	keys [][]byte
}

func newAPIKeys(keys []string) *apiKeys {
	if len(keys) == 0 {
		return nil
	}
	out := &apiKeys{}
	for _, key := range keys {
		out.keys = append(out.keys, []byte(key))
	}
	return out
}

// authorize reports whether r carries a valid key, removing the credential
// headers when it does.
func (k *apiKeys) authorize(r *http.Request) bool {
	if k == nil {
		return true
	}
	token, ok := bearerToken(r.Header.Get("Authorization"))
	if !ok || !k.valid(token) {
		token = r.Header.Get("Api-Key")
	}
	if token == "" || !k.valid(token) {
		return false
	}
	r.Header.Del("Authorization")
	r.Header.Del("Api-Key")
	return true
}

func (k *apiKeys) valid(token string) bool {
	found := 0
	for _, key := range k.keys {
		found |= subtle.ConstantTimeCompare(key, []byte(token))
	}
	return found == 1
}

// bearerToken matches the Lua pattern "^[Bb]earer%s+(.+)$".
func bearerToken(authorization string) (string, bool) {
	if len(authorization) < 7 || (authorization[0] != 'B' && authorization[0] != 'b') || authorization[1:6] != "earer" {
		return "", false
	}
	rest := strings.TrimLeft(authorization[6:], " \t\n\v\f\r")
	if len(rest) == len(authorization)-6 || rest == "" {
		return "", false
	}
	return rest, true
}

func writeUnauthorized(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("WWW-Authenticate", `Bearer realm="vllm-semantic-router"`)
	w.WriteHeader(http.StatusUnauthorized)
	_, _ = w.Write([]byte(unauthorizedBody))
}
