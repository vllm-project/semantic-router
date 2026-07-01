// Package imageurl provides shared validation for inline image inputs accepted
// across the router surfaces (ExtProc request path and the HTTP classification
// API). Keeping the gate in one place avoids the security logic drifting
// between callers.
package imageurl

import "strings"

// IsSafeImageDataURL returns true only for inline base64-encoded image data URIs
// with an allowlisted MIME type (e.g. "data:image/png;base64,...").
// HTTP(S) URLs, non-image data URIs, and file paths are rejected to prevent
// SSRF, local file access, and decode errors on non-image payloads.
func IsSafeImageDataURL(url string) bool {
	if url == "" {
		return false
	}
	lower := strings.ToLower(url)
	if !strings.HasPrefix(lower, "data:image/") {
		return false
	}
	const base64Sep = ";base64,"
	sepIdx := strings.Index(lower, base64Sep)
	if sepIdx == -1 {
		return false
	}
	mime := lower[len("data:"):sepIdx]
	switch mime {
	case "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp":
	default:
		return false
	}
	payload := strings.TrimSpace(url[sepIdx+len(base64Sep):])
	return payload != ""
}
