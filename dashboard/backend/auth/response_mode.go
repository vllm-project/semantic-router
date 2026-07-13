package auth

import (
	"errors"
	"net/http"
	"strings"
)

const (
	authResponseModeHeader = "X-VSR-Auth-Mode"
	authResponseModeCookie = "cookie"
	authResponseModeBearer = "bearer"
)

var errInvalidAuthResponseMode = errors.New("invalid authentication response mode")

// cookieOnlyAuthResponse defaults every login-like response to an HttpOnly
// cookie. A non-browser client must explicitly request the legacy bearer JSON
// contract, and browser request metadata makes that opt-in invalid. Ambiguous
// or unknown values fail closed instead of silently disclosing a token.
func cookieOnlyAuthResponse(r *http.Request) (bool, error) {
	if r == nil {
		return true, nil
	}
	values := r.Header.Values(authResponseModeHeader)
	if len(values) == 0 {
		return true, nil
	}
	if len(values) != 1 {
		return false, errInvalidAuthResponseMode
	}
	switch strings.TrimSpace(values[0]) {
	case authResponseModeCookie:
		return true, nil
	case authResponseModeBearer:
		if requestHasBrowserMetadata(r) {
			return false, errInvalidAuthResponseMode
		}
		return false, nil
	default:
		return false, errInvalidAuthResponseMode
	}
}

func requestHasBrowserMetadata(r *http.Request) bool {
	return len(r.Header.Values("Origin")) > 0 ||
		len(r.Header.Values("Sec-Fetch-Site")) > 0 ||
		len(r.Header.Values("Sec-Fetch-Mode")) > 0 ||
		len(r.Header.Values("Sec-Fetch-Dest")) > 0
}

func loginResponse(token string, user *User, cookieOnly bool) LoginResponse {
	response := LoginResponse{User: user}
	if !cookieOnly {
		response.Token = token
	}
	return response
}
