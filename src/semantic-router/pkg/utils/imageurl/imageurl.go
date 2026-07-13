// Package imageurl provides shared validation for inline image inputs accepted
// across the router surfaces (ExtProc request path and the HTTP classification
// API). Keeping the gate in one place avoids the security logic drifting
// between callers.
package imageurl

import (
	"bytes"
	"encoding/base64"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"strings"
)

const (
	dataPrefix                  = "data:"
	base64HeaderSuffix          = ";base64"
	base64Marker                = base64HeaderSuffix + ","
	maxEncodedImageBytes        = 20 * 1024 * 1024
	maxImageDimension           = 8192
	maxImagePixels       uint64 = 16_777_216
	// MaxImagePartsPerRequest bounds repeated full image decodes and native
	// inference calls within one classify, eval, or embeddings request.
	MaxImagePartsPerRequest = 8
	// MaxImagePixelsPerRequest bounds aggregate decompression work across all
	// accepted image parts. It permits eight full-HD images or one maximum-size
	// image while preventing a small compressed request from expanding repeatedly.
	MaxImagePixelsPerRequest uint64 = maxImagePixels
)

// ValidatedImage describes an inline image that passed the strict JPEG/PNG
// capability contract.
type ValidatedImage struct {
	DataURL      string
	Width        int
	Height       int
	Pixels       uint64
	EncodedBytes int
}

// RequestImageBudget tracks request-scoped image decode work. Its zero value is
// ready for use and must be shared across every image part in one request.
type RequestImageBudget struct {
	parts  int
	pixels uint64
}

func (b *RequestImageBudget) reservePart() bool {
	if b == nil {
		return true
	}
	if b.parts >= MaxImagePartsPerRequest {
		return false
	}
	b.parts++
	return true
}

func (b *RequestImageBudget) reservePixels(pixels uint64) bool {
	if b == nil {
		return true
	}
	if b.pixels > MaxImagePixelsPerRequest || pixels > MaxImagePixelsPerRequest-b.pixels {
		return false
	}
	b.pixels += pixels
	return true
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
	comma := strings.IndexByte(url, ',')
	if comma <= len(dataPrefix)+len(base64HeaderSuffix) {
		return "", "", false
	}
	header := url[:comma]
	if !strings.EqualFold(header[:len(dataPrefix)], dataPrefix) ||
		!strings.EqualFold(header[len(header)-len(base64HeaderSuffix):], base64HeaderSuffix) {
		return "", "", false
	}
	mime, ok = canonicalAllowedMIME(header[len(dataPrefix) : len(header)-len(base64HeaderSuffix)])
	if !ok {
		return "", "", false
	}
	payload = strings.TrimSpace(url[comma+1:])
	if payload == "" {
		return "", "", false
	}
	return mime, payload, true
}

func canonicalAllowedMIME(candidate string) (string, bool) {
	for _, allowed := range [...]string{"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"} {
		if strings.EqualFold(candidate, allowed) {
			return allowed, true
		}
	}
	return "", false
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

// IsJPEGOrPNGDataURLShape is a cheap capability-shape check. It does not
// decode base64 or inspect image bytes; callers must use
// ValidateJPEGOrPNGDataURL after expensive-work admission succeeds.
func IsJPEGOrPNGDataURLShape(url string) bool {
	mime, _, ok := parse(url)
	return ok && (mime == "image/png" || mime == "image/jpeg" || mime == "image/jpg")
}

// ValidateJPEGOrPNGDataURL applies the complete image capability contract and
// returns a canonical data URI on success. In addition to the generic data-URI
// shape gate, it bounds encoded bytes and decoded geometry, verifies that the
// declared MIME matches the detected JPEG/PNG format, and performs a full
// decode so truncated or otherwise corrupt images cannot reach native code. A
// non-nil budget is charged from DecodeConfig metadata before the full decode.
//
// IsSafeImageDataURL deliberately retains the broader GIF/WebP allowlist used
// by ExtProc. Callers that feed Candle's JPEG/PNG decoder must use this stricter
// validator.
func ValidateJPEGOrPNGDataURL(url string, budget *RequestImageBudget) (ValidatedImage, bool) {
	validated, data, expectedFormat, ok := inspectJPEGOrPNGDataURL(url, budget)
	if !ok || !fullyDecodesAs(data, expectedFormat, image.Config{
		Width: validated.Width, Height: validated.Height,
	}) {
		return ValidatedImage{}, false
	}
	return validated, true
}

// InspectJPEGOrPNGDataURL applies the bounded data-URI, base64, MIME/format,
// dimension, pixel-area, and aggregate request-budget checks without a full
// pixel decode. Use it only when the next trusted layer performs the sole full
// JPEG/PNG decode and reports decode failures as invalid client input.
func InspectJPEGOrPNGDataURL(url string, budget *RequestImageBudget) (ValidatedImage, bool) {
	validated, _, _, ok := inspectJPEGOrPNGDataURL(url, budget)
	return validated, ok
}

func inspectJPEGOrPNGDataURL(
	url string,
	budget *RequestImageBudget,
) (ValidatedImage, []byte, string, bool) {
	mime, payload, ok := parse(url)
	if !ok {
		return ValidatedImage{}, nil, "", false
	}
	expectedFormat, ok := expectedFormatForMIME(mime)
	if !ok || !budget.reservePart() {
		return ValidatedImage{}, nil, "", false
	}
	data, ok := decodeBoundedPayload(payload)
	if !ok {
		return ValidatedImage{}, nil, "", false
	}
	config, pixels, ok := inspectImageMetadata(data, expectedFormat)
	if !ok || !budget.reservePixels(pixels) {
		return ValidatedImage{}, nil, "", false
	}
	return ValidatedImage{
		DataURL:      "data:" + mime + base64Marker + payload,
		Width:        config.Width,
		Height:       config.Height,
		Pixels:       pixels,
		EncodedBytes: len(data),
	}, data, expectedFormat, true
}

func expectedFormatForMIME(mime string) (string, bool) {
	switch mime {
	case "image/png":
		return "png", true
	case "image/jpeg", "image/jpg":
		return "jpeg", true
	default:
		return "", false
	}
}

func decodeBoundedPayload(payload string) ([]byte, bool) {
	decoder := base64.NewDecoder(base64.StdEncoding, strings.NewReader(payload))
	data, err := io.ReadAll(io.LimitReader(decoder, maxEncodedImageBytes+1))
	return data, err == nil && len(data) > 0 && len(data) <= maxEncodedImageBytes
}

func inspectImageMetadata(data []byte, expectedFormat string) (image.Config, uint64, bool) {
	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || format != expectedFormat || !validDimensions(config.Width, config.Height) {
		return image.Config{}, 0, false
	}
	// #nosec G115 -- validDimensions proved both values positive and <= 8192.
	pixels := uint64(config.Width) * uint64(config.Height)
	return config, pixels, true
}

func fullyDecodesAs(data []byte, expectedFormat string, config image.Config) bool {
	decoded, format, err := image.Decode(bytes.NewReader(data))
	if err != nil || format != expectedFormat {
		return false
	}
	bounds := decoded.Bounds()
	return validDimensions(bounds.Dx(), bounds.Dy()) &&
		bounds.Dx() == config.Width && bounds.Dy() == config.Height
}

func validDimensions(width, height int) bool {
	if width <= 0 || height <= 0 || width > maxImageDimension || height > maxImageDimension {
		return false
	}
	// #nosec G115 -- the bounds above prove both values positive and <= 8192.
	pixels := uint64(width) * uint64(height)
	return pixels <= maxImagePixels
}

// DecodeBase64 validates that url is a generically safe image data URI whose
// payload is well-formed standard base64 and returns the decoded bytes. Candle
// consumers must use ValidateJPEGOrPNGDataURL, which also applies resource,
// MIME/format, and full-decode checks.
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
