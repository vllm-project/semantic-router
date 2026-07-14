package proxy

import (
	"net/http"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

const (
	dashboardSessionCookieName = "vsr_session"
	dashboardAuthQueryName     = "authToken"
)

func stripDashboardCredentials(r *http.Request, forwardAuthorization bool) {
	ac, authenticated := auth.AuthFromContext(r)
	if !forwardAuthorization || (authenticated && ac.CredentialSource == auth.CredentialSourceBearer) {
		r.Header.Del("Authorization")
	}
	r.Header.Del("Referer")
	r.URL.RawQuery = stripDashboardAuthQuery(r.URL.RawQuery)
	stripDashboardSessionCookies(r.Header)
}

func stripDashboardAuthQuery(rawQuery string) string {
	if rawQuery == "" {
		return ""
	}

	parts := strings.Split(rawQuery, "&")
	kept := parts[:0]
	for _, part := range parts {
		encodedName, _, _ := strings.Cut(part, "=")
		name, err := url.QueryUnescape(encodedName)
		if err == nil && name == dashboardAuthQueryName {
			continue
		}
		kept = append(kept, part)
	}
	return strings.Join(kept, "&")
}

func stripDashboardSessionCookies(header http.Header) {
	values := header.Values("Cookie")
	header.Del("Cookie")
	for _, value := range values {
		kept := make([]string, 0, strings.Count(value, ";")+1)
		for _, cookie := range strings.Split(value, ";") {
			cookie = strings.TrimSpace(cookie)
			if cookie == "" {
				continue
			}
			name, _, hasValue := strings.Cut(cookie, "=")
			if hasValue && strings.TrimSpace(name) == dashboardSessionCookieName {
				continue
			}
			kept = append(kept, cookie)
		}
		if len(kept) > 0 {
			header.Add("Cookie", strings.Join(kept, "; "))
		}
	}
}

func stripDashboardSessionSetCookies(header http.Header) {
	values := header.Values("Set-Cookie")
	header.Del("Set-Cookie")
	for _, value := range values {
		nameValue, _, _ := strings.Cut(value, ";")
		name, _, hasValue := strings.Cut(nameValue, "=")
		if hasValue && strings.TrimSpace(name) == dashboardSessionCookieName {
			continue
		}
		header.Add("Set-Cookie", value)
	}
}
