package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"net/url"
	"strings"
)

const (
	// Not HttpOnly on purpose: the frontend must read it to echo csrfHeaderName back.
	csrfCookieName = "vsr_csrf"
	csrfHeaderName = "X-CSRF-Token"

	// Domain-separates this HMAC from other uses of jwtSecret.
	csrfDerivationLabel = "csrf:"
)

// csrfTokenFor derives the CSRF token for a session, so nothing has to be stored and a
// planted cookie proves nothing. Returns "" so callers fail closed.
func csrfTokenFor(secret []byte, sessionID string) string {
	sessionID = strings.TrimSpace(sessionID)
	if len(secret) == 0 || sessionID == "" {
		return ""
	}
	mac := hmac.New(sha256.New, secret)
	mac.Write([]byte(csrfDerivationLabel + sessionID))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

// csrfTokenValid must stay constant time; "==" would leak the expected value.
func csrfTokenValid(secret []byte, sessionID, presented string) bool {
	expected := csrfTokenFor(secret, sessionID)
	if expected == "" || presented == "" {
		return false
	}
	return hmac.Equal([]byte(expected), []byte(presented))
}

// requestOrigin returns the origin this request was addressed to.
//
// SECURITY: X-Forwarded-Host is attacker-settable on a directly-reachable deployment, so
// it counts only when no allowlist is configured — in that mode this check is advisory and
// csrfTokenValid is the guarantee. Set DASHBOARD_ALLOWED_ORIGINS to make it load-bearing.
func requestOrigin(r *http.Request, trustForwardedHost bool) string {
	if r == nil {
		return ""
	}
	scheme := "http"
	if requestUsesHTTPS(r) {
		scheme = "https"
	}
	host := r.Host
	if trustForwardedHost {
		if forwarded := strings.TrimSpace(r.Header.Get("X-Forwarded-Host")); forwarded != "" {
			if comma := strings.Index(forwarded, ","); comma >= 0 {
				forwarded = strings.TrimSpace(forwarded[:comma])
			}
			host = forwarded
		}
	}
	if strings.TrimSpace(host) == "" {
		return ""
	}
	return strings.ToLower(scheme + "://" + host)
}

// originAllowed reports whether a state-changing request came from our own origin or one
// in allowed, which exists because the browser's origin and our Host legitimately differ
// behind a proxy and under the Vite dev proxy. Fails closed when neither header is present.
func originAllowed(r *http.Request, allowed []string) bool {
	if r == nil {
		return false
	}

	candidates := make([]string, 0, len(allowed)+1)
	if own := requestOrigin(r, len(allowed) == 0); own != "" {
		candidates = append(candidates, own)
	}
	for _, entry := range allowed {
		if entry = strings.ToLower(strings.TrimSpace(entry)); entry != "" {
			candidates = append(candidates, entry)
		}
	}
	if len(candidates) == 0 {
		return false
	}

	presented := ""
	if origin := strings.TrimSpace(r.Header.Get("Origin")); origin != "" {
		// An opaque origin sends "null"; it must never match, even if listed.
		if strings.EqualFold(origin, "null") {
			return false
		}
		presented = strings.ToLower(origin)
	} else if referer := strings.TrimSpace(r.Header.Get("Referer")); referer != "" {
		parsed, err := url.Parse(referer)
		if err != nil || parsed.Scheme == "" || parsed.Host == "" {
			return false
		}
		presented = strings.ToLower(parsed.Scheme + "://" + parsed.Host)
	}
	if presented == "" {
		return false
	}

	for _, candidate := range candidates {
		if presented == candidate {
			return true
		}
	}
	return false
}

// OriginChecker returns a CheckOrigin function for the WebSocket upgrader, which cannot
// rely on CORS: a handshake is exempt from it, so this is the only origin control a
// WebSocket server has. Takes the allowlist as a closure so the caller needs no *Service;
// with a nil list it accepts our own origin and rejects genuinely cross-origin handshakes.
//
// A handshake carrying no Origin is accepted, because RFC 6455 requires browser clients to
// send one: its absence means a non-browser client (a Go dialer, curl, CI), which no
// hostile page can drive cross-site. Same reasoning as the Authorization: Bearer exemption,
// and such a client still has to authenticate.
func OriginChecker(allowedOrigins []string) func(*http.Request) bool {
	return func(r *http.Request) bool {
		if r == nil {
			return false
		}
		if strings.TrimSpace(r.Header.Get("Origin")) == "" {
			return true
		}
		return originAllowed(r, allowedOrigins)
	}
}

// requiresCSRFCheck exempts the RFC 9110 safe methods, which holds only while no GET route
// mutates state.
func requiresCSRFCheck(method string) bool {
	switch strings.ToUpper(strings.TrimSpace(method)) {
	case http.MethodGet, http.MethodHead, http.MethodOptions:
		return false
	default:
		return true
	}
}
