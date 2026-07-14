package auth

import (
	"errors"
	"net/http"
	"net/url"
	"strings"
	"unicode"

	"github.com/vllm-project/semantic-router/dashboard/backend/browsersecurity"
)

const (
	authSessionCookieName = "vsr_session"
	maxAccessTokenBytes   = 8192
)

type CredentialSource string

var (
	errQueryCredential     = errors.New("access tokens in URL query parameters are forbidden")
	errAmbiguousCredential = errors.New("multiple credential transports are forbidden")
)

const (
	CredentialSourceBearer CredentialSource = "bearer"
	CredentialSourceCookie CredentialSource = "cookie"
)

func extractBearer(raw string) string {
	if raw == "" {
		return ""
	}
	parts := strings.SplitN(raw, " ", 2)
	if len(parts) != 2 {
		return ""
	}
	if !strings.EqualFold(parts[0], "bearer") {
		return ""
	}
	return normalizeAccessToken(parts[1])
}

func extractAccessToken(r *http.Request) string {
	token, _ := extractAccessTokenWithSource(r)
	return token
}

func extractAccessTokenWithSource(r *http.Request) (string, CredentialSource) {
	if token := extractBearer(r.Header.Get("Authorization")); token != "" {
		return token, CredentialSourceBearer
	}

	if cookie, err := r.Cookie(authSessionCookieName); err == nil {
		if token := normalizeAccessToken(cookie.Value); token != "" {
			return token, CredentialSourceCookie
		}
	}

	return "", ""
}

// validateCredentialTransport rejects token transports that can leak through
// browser history/referrers and requests whose identity depends on credential
// precedence. Callers must run this before routing, proxying, or request logs.
func validateCredentialTransport(r *http.Request) error {
	if r == nil || r.URL == nil {
		return nil
	}
	if containsQueryCredential(r.URL.RawQuery) {
		return errQueryCredential
	}

	authorizationValues := r.Header.Values("Authorization")
	if len(authorizationValues) > 1 {
		return errAmbiguousCredential
	}
	hasAuthorization := false
	for _, value := range authorizationValues {
		if strings.TrimSpace(value) != "" {
			hasAuthorization = true
			break
		}
	}
	hasSessionCookie := false
	sessionCookieCount := 0
	for _, cookie := range r.Cookies() {
		if cookie.Name != authSessionCookieName {
			continue
		}
		sessionCookieCount++
		if sessionCookieCount > 1 {
			return errAmbiguousCredential
		}
		if strings.TrimSpace(cookie.Value) != "" {
			hasSessionCookie = true
		}
	}
	if hasAuthorization && hasSessionCookie {
		return errAmbiguousCredential
	}
	return nil
}

func containsQueryCredential(rawQuery string) bool {
	for _, field := range strings.Split(rawQuery, "&") {
		rawKey, _, _ := strings.Cut(field, "=")
		key, err := url.QueryUnescape(rawKey)
		if err != nil {
			key = rawKey
		}
		if strings.EqualFold(strings.TrimSpace(key), "authToken") {
			return true
		}
	}
	return false
}

func normalizeAccessToken(raw string) string {
	token := strings.TrimSpace(raw)
	if token == "" || len(token) > maxAccessTokenBytes {
		return ""
	}
	for _, r := range token {
		if r == ';' || unicode.IsControl(r) || unicode.IsSpace(r) {
			return ""
		}
	}
	return token
}

func validUnsafeRequestOrigin(r *http.Request, credentialSource CredentialSource) bool {
	switch r.Method {
	case http.MethodGet, http.MethodHead, http.MethodOptions:
		return true
	}

	if len(r.Header.Values("Origin")) > 0 {
		return browsersecurity.ValidOrigin(r)
	}
	fetchSiteValues := r.Header.Values("Sec-Fetch-Site")
	if len(fetchSiteValues) > 1 {
		return false
	}
	if len(fetchSiteValues) == 1 {
		return strings.EqualFold(strings.TrimSpace(fetchSiteValues[0]), "same-origin")
	}

	// Non-browser API clients can continue using bearer authentication without
	// browser Origin metadata. Cookie credentials require same-origin browser
	// evidence on every unsafe method. Access tokens in URI query parameters are
	// never accepted because URLs routinely escape into history, logs, and
	// referrers.
	return credentialSource == CredentialSourceBearer
}
