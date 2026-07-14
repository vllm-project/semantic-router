package browsersecurity

import (
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

const forwardedProtoHeader = "X-Forwarded-Proto"

// ValidOrigin reports whether a browser Origin is exactly the effective
// request origin. It rejects missing, repeated, ambiguous, and malformed proxy
// metadata instead of treating registrable-domain siblings as equivalent.
func ValidOrigin(r *http.Request) bool {
	originValues := r.Header.Values("Origin")
	if len(originValues) != 1 {
		return false
	}
	origin := originValues[0]
	if origin == "" || origin != strings.TrimSpace(origin) || strings.EqualFold(origin, "null") {
		return false
	}

	originURL, err := url.Parse(origin)
	if err != nil || originURL.Opaque != "" || originURL.User != nil ||
		originURL.Path != "" || originURL.RawPath != "" ||
		originURL.RawQuery != "" || originURL.Fragment != "" {
		return false
	}
	originScheme := strings.ToLower(originURL.Scheme)
	if originScheme != "http" && originScheme != "https" {
		return false
	}

	requestScheme, ok := effectiveRequestScheme(r)
	if !ok || originScheme != requestScheme {
		return false
	}
	originAuthority, ok := canonicalAuthority(originURL.Host, originScheme)
	if !ok {
		return false
	}
	requestAuthority, ok := canonicalAuthority(r.Host, requestScheme)
	return ok && originAuthority == requestAuthority
}

func effectiveRequestScheme(r *http.Request) (string, bool) {
	forwardedValues := r.Header.Values(forwardedProtoHeader)
	if len(forwardedValues) > 1 {
		return "", false
	}

	forwardedScheme := ""
	if len(forwardedValues) == 1 {
		forwardedScheme = strings.ToLower(strings.TrimSpace(forwardedValues[0]))
		if forwardedScheme == "" || strings.Contains(forwardedScheme, ",") ||
			(forwardedScheme != "http" && forwardedScheme != "https") {
			return "", false
		}
	}

	if r.TLS != nil {
		if forwardedScheme != "" && forwardedScheme != "https" {
			return "", false
		}
		return "https", true
	}
	if forwardedScheme != "" {
		return forwardedScheme, true
	}
	return "http", true
}

func canonicalAuthority(authority string, scheme string) (string, bool) {
	if authority == "" || authority != strings.TrimSpace(authority) ||
		strings.ContainsAny(authority, "/?#@") {
		return "", false
	}
	parsed, err := url.Parse("//" + authority)
	if err != nil || parsed.User != nil || parsed.Host == "" || parsed.Path != "" ||
		parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", false
	}

	hostname := strings.ToLower(parsed.Hostname())
	if hostname == "" || strings.Contains(hostname, "%") {
		return "", false
	}
	port := parsed.Port()
	if port == "" {
		if scheme == "https" {
			port = "443"
		} else {
			port = "80"
		}
	}
	portNumber, err := strconv.Atoi(port)
	if err != nil || portNumber < 1 || portNumber > 65535 {
		return "", false
	}
	return net.JoinHostPort(hostname, strconv.Itoa(portNumber)), true
}
