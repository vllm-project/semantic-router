package cache

import "strings"

// normalizeLocalHostForContainerRuntimes forces IPv4 loopback for localhost.
// In some rootless/containerized local environments, "localhost" can resolve to
// "::1" and produce EOF/reset errors when services only listen on IPv4.
func normalizeLocalHostForContainerRuntimes(host string) string {
	normalized := strings.TrimSpace(host)
	if strings.EqualFold(normalized, "localhost") {
		return "127.0.0.1"
	}
	return normalized
}
