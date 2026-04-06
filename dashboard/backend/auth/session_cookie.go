package auth

import (
	"net/http"
	"strings"
	"time"
)

const authSessionCookiePath = "/"

func writeSessionCookie(w http.ResponseWriter, r *http.Request, svc *Service, token string) {
	if strings.TrimSpace(token) == "" {
		return
	}

	http.SetCookie(w, &http.Cookie{
		Name:     authSessionCookieName,
		Value:    token,
		Path:     authSessionCookiePath,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   requestUsesHTTPS(r),
		MaxAge:   int(svc.ttlDuration.Seconds()),
		Expires:  time.Now().Add(svc.ttlDuration),
	})
}

func clearSessionCookie(w http.ResponseWriter, r *http.Request) {
	http.SetCookie(w, &http.Cookie{
		Name:     authSessionCookieName,
		Value:    "",
		Path:     authSessionCookiePath,
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   requestUsesHTTPS(r),
		MaxAge:   -1,
		Expires:  time.Unix(0, 0),
	})
}

func requestUsesHTTPS(r *http.Request) bool {
	if r == nil {
		return false
	}
	if r.TLS != nil {
		return true
	}

	forwardedProto := strings.TrimSpace(r.Header.Get("X-Forwarded-Proto"))
	if forwardedProto == "" {
		return false
	}

	parts := strings.Split(forwardedProto, ",")
	return strings.EqualFold(strings.TrimSpace(parts[0]), "https")
}
