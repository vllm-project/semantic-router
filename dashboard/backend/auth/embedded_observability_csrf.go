package auth

import (
	"mime"
	"net/http"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/observability"
)

// Grafana's own frontend does not know the Dashboard's session-bound CSRF
// header. Its data-source query API uses POST for reads. Only that exact API
// may instead prove browser intent with a same-origin JSON request; dashboard
// and Grafana mutations still require the normal CSRF token. Authentication and
// permission checks always run, including for this read-only POST.
func embeddedGrafanaQueryAllowed(r *http.Request, allowedOrigins []string) bool {
	if r == nil || r.Method != http.MethodPost || !observability.IsGrafanaQueryPath(r.URL.Path) {
		return false
	}
	// An invalid explicitly supplied token must not fall back to this policy.
	if len(r.Header.Values(csrfHeaderName)) != 0 {
		return false
	}
	contentType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || contentType != "application/json" {
		return false
	}
	// Fetch Metadata is browser-controlled. Same-site is insufficient, and
	// older clients without it must still supply the explicit Origin below.
	if site := r.Header.Get("Sec-Fetch-Site"); site != "" && site != "same-origin" {
		return false
	}
	origin, err := url.Parse(strings.TrimSpace(r.Header.Get("Origin")))
	if err != nil || origin.Host == "" || origin.User != nil || origin.Path != "" ||
		origin.RawQuery != "" || origin.Fragment != "" ||
		(origin.Scheme != "http" && origin.Scheme != "https") {
		return false
	}
	// Unlike the token-protected policy, never derive this trust decision from
	// a caller-supplied X-Forwarded-Host. Operators may configure extra origins.
	return originAllowedAgainst(r, allowedOrigins, requestOrigin(r, false))
}
