package proxy

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"unicode"
)

var reservedWebSocketStaticHeaders = map[string]struct{}{
	"connection":            {},
	"content-length":        {},
	"host":                  {},
	"keep-alive":            {},
	"proxy-authenticate":    {},
	"proxy-authorization":   {},
	"proxy-connection":      {},
	"sec-websocket-key":     {},
	"sec-websocket-version": {},
	"te":                    {},
	"trailer":               {},
	"transfer-encoding":     {},
	"upgrade":               {},
}

func normalizeStaticHeaders(staticHeaders map[string]string) (map[string]string, error) {
	normalized := make(map[string]string, len(staticHeaders))
	for key, value := range staticHeaders {
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if key == "" || value == "" {
			continue
		}
		if !validHTTPHeaderName(key) || !validHTTPHeaderValue(value) ||
			isReservedWebSocketStaticHeader(key) {
			return nil, fmt.Errorf("invalid static proxy header")
		}
		normalized[http.CanonicalHeaderKey(key)] = value
	}
	return normalized, nil
}

func isReservedWebSocketStaticHeader(name string) bool {
	_, reserved := reservedWebSocketStaticHeaders[strings.ToLower(strings.TrimSpace(name))]
	return reserved
}

func validHTTPHeaderName(name string) bool {
	for i := 0; i < len(name); i++ {
		c := name[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') {
			continue
		}
		switch c {
		case '!', '#', '$', '%', '&', '\'', '*', '+', '-', '.', '^', '_', '`', '|', '~':
			continue
		default:
			return false
		}
	}
	return name != ""
}

func validHTTPHeaderValue(value string) bool {
	for i := 0; i < len(value); i++ {
		if value[i] == '\r' || value[i] == '\n' || value[i] == 0 || value[i] == 0x7f ||
			(value[i] < 0x20 && value[i] != '\t') {
			return false
		}
	}
	for _, r := range value {
		if unicode.IsControl(r) && r != '\t' {
			return false
		}
	}
	return true
}

func buildWebSocketUpgradeRequest(
	r *http.Request,
	target *url.URL,
	path string,
	staticHeaders map[string]string,
) ([]byte, error) {
	requestURL, err := webSocketRequestTarget(path, r.URL.RawQuery)
	if err != nil {
		return nil, err
	}
	if r.Method != http.MethodGet || target.Host == "" || !validHTTPHeaderValue(target.Host) {
		return nil, fmt.Errorf("invalid WebSocket upgrade metadata")
	}

	var request strings.Builder
	request.WriteString(r.Method + " " + requestURL + " HTTP/1.1\r\n")
	request.WriteString("Host: " + target.Host + "\r\n")
	request.WriteString("Connection: Upgrade\r\n")
	request.WriteString("Upgrade: websocket\r\n")
	if err := appendWebSocketRequestHeaders(&request, r.Header, staticHeaders); err != nil {
		return nil, err
	}
	for key, value := range staticHeaders {
		if !validHTTPHeaderName(key) || !validHTTPHeaderValue(value) {
			return nil, fmt.Errorf("invalid WebSocket upgrade header")
		}
		request.WriteString(key + ": " + value + "\r\n")
	}
	request.WriteString("\r\n")
	return []byte(request.String()), nil
}

func webSocketRequestTarget(path, rawQuery string) (string, error) {
	if !strings.HasPrefix(path, "/") {
		return "", fmt.Errorf("invalid WebSocket request target")
	}
	for _, r := range path {
		if unicode.IsControl(r) {
			return "", fmt.Errorf("invalid WebSocket request target")
		}
	}

	requestURL := (&url.URL{
		Path:     path,
		RawQuery: stripDashboardAuthQuery(rawQuery),
	}).RequestURI()
	for i := 0; i < len(requestURL); i++ {
		if requestURL[i] <= 0x20 || requestURL[i] == 0x7f {
			return "", fmt.Errorf("invalid WebSocket request target")
		}
	}
	return requestURL, nil
}

func appendWebSocketRequestHeaders(
	request *strings.Builder,
	header http.Header,
	staticHeaders map[string]string,
) error {
	overriddenHeaders := normalizedHeaderKeySet(staticHeaders)
	connectionHeaders := connectionNominatedHeaders(header.Values("Connection"))
	for key, values := range header {
		if shouldDropWebSocketHeader(key, overriddenHeaders, connectionHeaders) {
			continue
		}
		if !validHTTPHeaderName(key) {
			return fmt.Errorf("invalid WebSocket upgrade header")
		}
		if strings.EqualFold(key, "Cookie") {
			cookieHeader := http.Header{"Cookie": append([]string(nil), values...)}
			stripDashboardSessionCookies(cookieHeader)
			values = cookieHeader.Values("Cookie")
		}
		for _, value := range values {
			if !validHTTPHeaderValue(value) {
				return fmt.Errorf("invalid WebSocket upgrade header")
			}
			request.WriteString(key + ": " + value + "\r\n")
		}
	}
	return nil
}

func connectionNominatedHeaders(values []string) map[string]struct{} {
	headers := make(map[string]struct{})
	for _, value := range values {
		for _, token := range strings.Split(value, ",") {
			token = strings.ToLower(strings.TrimSpace(token))
			if token != "" {
				headers[token] = struct{}{}
			}
		}
	}
	return headers
}

func normalizedHeaderKeySet(headers map[string]string) map[string]struct{} {
	keys := make(map[string]struct{}, len(headers))
	for key := range headers {
		key = strings.ToLower(strings.TrimSpace(key))
		if key != "" {
			keys[key] = struct{}{}
		}
	}
	return keys
}

func shouldDropWebSocketHeader(
	key string,
	overriddenHeaders map[string]struct{},
	connectionHeaders map[string]struct{},
) bool {
	normalizedKey := strings.ToLower(strings.TrimSpace(key))
	if normalizedKey == "host" || normalizedKey == "authorization" || normalizedKey == "referer" ||
		normalizedKey == "connection" || normalizedKey == "upgrade" ||
		normalizedKey == "proxy-authenticate" || normalizedKey == "proxy-authorization" ||
		normalizedKey == "proxy-connection" || normalizedKey == "keep-alive" ||
		normalizedKey == "te" || normalizedKey == "trailer" ||
		normalizedKey == "transfer-encoding" || normalizedKey == "content-length" {
		return true
	}
	if _, nominated := connectionHeaders[normalizedKey]; nominated {
		return true
	}
	_, overridden := overriddenHeaders[normalizedKey]
	return overridden
}
