package auth

import (
	"errors"
	"net/http"
	"strings"
)

const (
	authResponseModeHeader = "X-VSR-Auth-Mode"
	authResponseModeCookie = "cookie"
)

var errInvalidAuthResponseMode = errors.New("invalid authentication response mode")

// cookieOnlyAuthResponse reports whether a maintained browser client requested
// an HttpOnly-cookie-only response. Omitting the header preserves the existing
// bearer-token response contract for non-browser API clients. Ambiguous or
// unknown values fail closed instead of silently disclosing a token.
func cookieOnlyAuthResponse(r *http.Request) (bool, error) {
	if r == nil {
		return false, nil
	}
	values := r.Header.Values(authResponseModeHeader)
	if len(values) == 0 {
		return false, nil
	}
	if len(values) != 1 || strings.TrimSpace(values[0]) != authResponseModeCookie {
		return false, errInvalidAuthResponseMode
	}
	return true, nil
}

func loginResponse(token string, user *User, cookieOnly bool) LoginResponse {
	response := LoginResponse{User: user}
	if !cookieOnly {
		response.Token = token
	}
	return response
}
