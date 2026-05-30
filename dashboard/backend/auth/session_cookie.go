package auth

import (
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

		if token := extractAccessToken(r); token != "" && svc != nil {
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

	http.SetCookie(w, &http.Cookie{
		Name:     authSessionCookieName,
		Value:    token,
		Path:     "/",
		MaxAge:   int(effectiveTTL.Seconds()),
		Expires:  time.Now().Add(effectiveTTL),
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   requestUsesHTTPS(r),
	})
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
	return strings.EqualFold(strings.TrimSpace(r.Header.Get("X-Forwarded-Proto")), "https")
}
