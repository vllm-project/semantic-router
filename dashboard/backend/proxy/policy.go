package proxy

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

// ValidateDashboardOrigin rejects browser-originated proxy requests that do not
// come from the dashboard's own origin.
func ValidateDashboardOrigin(r *http.Request) error {
	if r == nil {
		return fmt.Errorf("request is required")
	}

	expectedOrigin := requestOrigin(r)
	if expectedOrigin == "" {
		return nil
	}

	if origin := normalizeOrigin(r.Header.Get("Origin")); origin != "" {
		if origin == expectedOrigin {
			return nil
		}
		return fmt.Errorf("cross-origin proxy access is not allowed")
	}

	if refererOrigin := normalizeOrigin(r.Header.Get("Referer")); refererOrigin != "" {
		if refererOrigin == expectedOrigin {
			return nil
		}
		return fmt.Errorf("cross-origin proxy access is not allowed")
	}

	fetchSite := strings.ToLower(strings.TrimSpace(r.Header.Get("Sec-Fetch-Site")))
	if fetchSite == "" || fetchSite == "same-origin" || fetchSite == "none" {
		return nil
	}

	return fmt.Errorf("cross-origin proxy access is not allowed")
}

func requestOrigin(r *http.Request) string {
	if r == nil {
		return ""
	}

	host := strings.TrimSpace(r.Host)
	if host == "" {
		host = strings.TrimSpace(r.Header.Get("X-Forwarded-Host"))
	}
	if host == "" {
		return ""
	}

	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	if forwardedProto := strings.TrimSpace(r.Header.Get("X-Forwarded-Proto")); forwardedProto != "" {
		scheme = strings.ToLower(strings.Split(forwardedProto, ",")[0])
	}

	return strings.ToLower(scheme + "://" + host)
}

func normalizeOrigin(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}

	parsed, err := url.Parse(trimmed)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return ""
	}

	return strings.ToLower(parsed.Scheme + "://" + parsed.Host)
}
