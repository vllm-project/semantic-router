// Package imageurl provides shared validation for inline image inputs accepted
// across the router surfaces (ExtProc request path and the HTTP classification
// API). Keeping the gate in one place avoids the security logic drifting
// between callers.
package imageurl

import (
	"encoding/base64"
	"strings"
)

const base64Marker = ";base64,"

var allowedMIME = map[string]bool{
	"image/png":  true,
	"image/jpeg": true,
	"image/jpg":  true,
	"image/gif":  true,
	"image/webp": true,
}

// parse is the single parser behind every exported helper. It returns the
// lowercased MIME type and the exact-case base64 payload: the "data:" scheme and
// ";base64," marker are matched case-insensitively (RFC 2397), but the payload is
// preserved verbatim because the base64 alphabet is case-sensitive. ok is false
// for anything that is not an allowlisted inline base64 image data URI.
func parse(url string) (mime string, payload string, ok bool) {
	if url == "" {
		return "", "", false
	}
	lower := strings.ToLower(url)
	if !strings.HasPrefix(lower, "data:image/") {
		return "", "", false
	}
	sepIdx := strings.Index(lower, base64Marker)
	if sepIdx == -1 {
		return "", "", false
	}
	mime = lower[len("data:"):sepIdx]
	if !allowedMIME[mime] {
		return "", "", false
	}
	payload = strings.TrimSpace(url[sepIdx+len(base64Marker):])
	if payload == "" {
		return "", "", false
	}
	return mime, payload, true
}

// IsSafeImageDataURL returns true only for inline base64-encoded image data URIs
// with an allowlisted MIME type (e.g. "data:image/png;base64,..."). HTTP(S) URLs,
// non-image data URIs, and file paths are rejected to prevent SSRF, local file
// access, and decode errors. It checks shape only; use DecodeBase64 to validate
// the payload.
func IsSafeImageDataURL(url string) bool {
	_, _, ok := parse(url)
	return ok
}

// DecodeBase64 validates that url is a safe image data URI whose payload is
// well-formed standard base64 and returns the decoded bytes. Request validation
// uses it so malformed input is rejected as a 400 rather than failing later at
// the FFI image decoder as a 500.
func DecodeBase64(url string) ([]byte, bool) {
	_, payload, ok := parse(url)
	if !ok {
		return nil, false
	}
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil || len(data) == 0 {
		return nil, false
	}
	return data, true
}

// CanonicalDataURL rebuilds the data URI with a lowercased scheme, MIME type, and
// ";base64," marker while preserving the exact-case payload, for consumers that
// scan for the marker case-sensitively (e.g. the candle FFI).
func CanonicalDataURL(url string) (string, bool) {
	mime, payload, ok := parse(url)
	if !ok {
		return "", false
	}
	return "data:" + mime + base64Marker + payload, true
}
