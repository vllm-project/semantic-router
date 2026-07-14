package auth

import (
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/browsersecurity"
)

const forwardedProtoHeader = "X-Forwarded-Proto"

// IsWebSocketUpgradeRequest recognizes the standard Connection/Upgrade token
// pair used by every protected dashboard WebSocket endpoint.
func IsWebSocketUpgradeRequest(r *http.Request) bool {
	connectionUpgrade := false
	for _, value := range r.Header.Values("Connection") {
		for _, token := range strings.Split(value, ",") {
			if strings.EqualFold(strings.TrimSpace(token), "upgrade") {
				connectionUpgrade = true
				break
			}
		}
	}
	if !connectionUpgrade {
		return false
	}
	for _, value := range r.Header.Values("Upgrade") {
		for _, token := range strings.Split(value, ",") {
			if strings.EqualFold(strings.TrimSpace(token), "websocket") {
				return true
			}
		}
	}
	return false
}

// ValidWebSocketOrigin prevents cross-origin WebSocket hijacking for
// cookie-authenticated dashboard connections. Browser clients must send one
// strict HTTP(S) Origin whose canonical authority and effective request scheme
// match the dashboard request. X-Forwarded-Host is intentionally not trusted.
func ValidWebSocketOrigin(r *http.Request) bool {
	return browsersecurity.ValidOrigin(r)
}
