package auth

import (
	"math"
	"net/http"
	"strings"
	"time"
)

const defaultAuthSessionCookieTTL = 12 * time.Hour

func logoutHandler(svc *Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		token, credentialSource := extractAccessTokenWithSource(r)
		// Logout always mutates browser state by expiring the session cookie,
		// even when SameSite prevents the cookie from reaching this request.
		// Only an explicit bearer client may omit browser origin metadata.
		if credentialSource != CredentialSourceBearer &&
			!validUnsafeRequestOrigin(r, credentialSource) {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}

		if token != "" && svc != nil {
			if err := svc.RevokeToken(r.Context(), token); err != nil {
				clearAuthSessionCookie(w, r)
				http.Error(w, "logout failed", http.StatusInternalServerError)
				return
			}
		}
		clearAuthSessionCookie(w, r)
		respondJSON(w, map[string]bool{"ok": true})
	}
}

func setAuthSessionCookie(w http.ResponseWriter, r *http.Request, token string, ttl time.Duration) {
	if strings.TrimSpace(token) == "" {
		return
	}

	effectiveTTL := authSessionCookieTTL(ttl)
	setAuthSessionCookieUntil(w, r, token, time.Now().Add(effectiveTTL))
}

func setAuthSessionCookieUntil(
	w http.ResponseWriter,
	r *http.Request,
	token string,
	expiresAt time.Time,
) {
	if strings.TrimSpace(token) == "" || expiresAt.IsZero() {
		return
	}
	remaining := time.Until(expiresAt)
	if remaining <= 0 {
		return
	}

	http.SetCookie(w, &http.Cookie{
		Name:     authSessionCookieName,
		Value:    token,
		Path:     "/",
		MaxAge:   int(math.Ceil(remaining.Seconds())),
		Expires:  expiresAt,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   requestUsesHTTPS(r),
	})
}

// reissueSessionCookie replaces a legacy JavaScript-created session cookie
// with the same server-validated JWT and hardened attributes. It does not mint
// a token, extend the JWT expiry, or create another server-side session.
func reissueSessionCookie(w http.ResponseWriter, r *http.Request, svc *Service) {
	if svc == nil {
		return
	}
	token := extractAccessToken(r)
	claims, err := svc.ParseToken(token)
	if err != nil || claims.ExpiresAt == nil {
		return
	}
	setAuthSessionCookieUntil(w, r, token, claims.ExpiresAt.Time)
}

func clearAuthSessionCookie(w http.ResponseWriter, r *http.Request) {
	http.SetCookie(w, &http.Cookie{
		Name:     authSessionCookieName,
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		Expires:  time.Unix(0, 0),
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   requestUsesHTTPS(r),
	})
}

func authSessionCookieTTL(ttl time.Duration) time.Duration {
	if ttl <= 0 {
		return defaultAuthSessionCookieTTL
	}
	return ttl
}

func requestUsesHTTPS(r *http.Request) bool {
	if r == nil {
		return false
	}
	if r.TLS != nil {
		return true
	}

	// A proxy chain appends its own protocol to X-Forwarded-Proto. The
	// left-most hop is the original client protocol; later HTTP hops must not
	// downgrade a cookie issued to an HTTPS client. As with all forwarded
	// metadata, deployments must overwrite untrusted client input at the edge.
	forwardedValues := r.Header.Values("X-Forwarded-Proto")
	if len(forwardedValues) == 0 {
		return false
	}
	firstHop, _, _ := strings.Cut(forwardedValues[0], ",")
	return strings.EqualFold(strings.TrimSpace(firstHop), "https")
}
