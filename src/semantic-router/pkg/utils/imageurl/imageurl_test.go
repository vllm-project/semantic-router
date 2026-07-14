package imageurl

import (
	"bytes"
	"encoding/base64"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"strings"
	"testing"
)

func testImageDataURL(t *testing.T, mime string) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.Set(0, 0, color.RGBA{R: 255, G: 64, B: 32, A: 255})
	var encoded bytes.Buffer
	var err error
	switch mime {
	case "image/png":
		err = png.Encode(&encoded, img)
	case "image/jpeg", "image/jpg":
		err = jpeg.Encode(&encoded, img, &jpeg.Options{Quality: 90})
	default:
		t.Fatalf("unsupported test MIME %q", mime)
	}
	if err != nil {
		t.Fatalf("encode %s fixture: %v", mime, err)
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
}

func pngWithDeclaredDimensions(t *testing.T, width, height uint32) string {
	t.Helper()
	dataURL := testImageDataURL(t, "image/png")
	payload := strings.TrimPrefix(dataURL, "data:image/png;base64,")
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		t.Fatalf("decode PNG fixture: %v", err)
	}
	data[16] = byte(width >> 24)
	data[17] = byte(width >> 16)
	data[18] = byte(width >> 8)
	data[19] = byte(width)
	data[20] = byte(height >> 24)
	data[21] = byte(height >> 16)
	data[22] = byte(height >> 8)
	data[23] = byte(height)
	checksum := crc32.ChecksumIEEE(data[12:29])
	data[29] = byte(checksum >> 24)
	data[30] = byte(checksum >> 16)
	data[31] = byte(checksum >> 8)
	data[32] = byte(checksum)
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(data)
}

func TestIsSafeImageDataURL(t *testing.T) {
	cases := []struct {
		name string
		url  string
		want bool
	}{
		{"png data uri", testImageDataURL(t, "image/png"), true},
		{"jpeg data uri", testImageDataURL(t, "image/jpeg"), true},
		{"uppercase mime", "DATA:IMAGE/WEBP;BASE64,abc", true},
		{"empty", "", false},
		{"http url", "https://example.com/cat.png", false},
		{"non-image data uri", "data:text/plain;base64,aGVsbG8=", false},
		{"missing base64 marker", "data:image/png,rawbytes", false},
		{"empty payload", "data:image/png;base64,", false},
		{"disallowed mime", "data:image/svg+xml;base64,abc", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsSafeImageDataURL(tc.url); got != tc.want {
				t.Fatalf("IsSafeImageDataURL(%q) = %v, want %v", tc.url, got, tc.want)
			}
		})
	}
}

func TestValidateJPEGOrPNGDataURL(t *testing.T) {
	pngURL := testImageDataURL(t, "image/png")
	jpegURL := testImageDataURL(t, "image/jpeg")
	jpgURL := testImageDataURL(t, "image/jpg")
	uppercasePNG := strings.Replace(pngURL, "data:image/png;base64,", "DATA:IMAGE/PNG;BASE64,", 1)

	truncatedPayload := strings.TrimPrefix(pngURL, "data:image/png;base64,")
	truncatedBytes, err := base64.StdEncoding.DecodeString(truncatedPayload)
	if err != nil {
		t.Fatalf("decode PNG fixture: %v", err)
	}
	truncatedURL := "data:image/png;base64," + base64.StdEncoding.EncodeToString(truncatedBytes[:len(truncatedBytes)-8])

	cases := []struct {
		name string
		url  string
		want bool
	}{
		{"png", pngURL, true},
		{"jpeg", jpegURL, true},
		{"jpg alias", jpgURL, true},
		{"uppercase canonicalized", uppercasePNG, true},
		{"valid base64 non-image", "data:image/png;base64,aGVsbG8=", false},
		{"truncated PNG", truncatedURL, false},
		{"declared JPEG containing PNG", strings.Replace(pngURL, "image/png", "image/jpeg", 1), false},
		{"declared PNG containing JPEG", strings.Replace(jpegURL, "image/jpeg", "image/png", 1), false},
		{"GIF remains safe but is not embeddable", "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==", false},
		{"WebP remains safe but is not embeddable", "data:image/webp;base64,UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEAAUAmJaQAA3AA/v89WAAAAA==", false},
		{"oversized dimension", pngWithDeclaredDimensions(t, maxImageDimension+1, 1), false},
		{"oversized pixel area", pngWithDeclaredDimensions(t, 4097, 4097), false},
		{"unsafe URL", "https://example.com/cat.png", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			validated, ok := ValidateJPEGOrPNGDataURL(tc.url, nil)
			if ok != tc.want {
				t.Fatalf("ValidateJPEGOrPNGDataURL(%q) ok = %v, want %v", tc.name, ok, tc.want)
			}
			if ok && !strings.HasPrefix(validated.DataURL, "data:image/") {
				t.Fatalf("unexpected canonical URI %q", validated.DataURL)
			}
		})
	}

	validated, ok := ValidateJPEGOrPNGDataURL(uppercasePNG, nil)
	if !ok || validated.DataURL != pngURL {
		t.Fatalf("uppercase URI canonicalized to %q, want %q", validated.DataURL, pngURL)
	}

	for _, url := range []string{
		"data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==",
		"data:image/webp;base64,UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEAAUAmJaQAA3AA/v89WAAAAA==",
	} {
		if !IsSafeImageDataURL(url) {
			t.Fatalf("generic data-URI allowlist must continue accepting %q", url)
		}
	}
}

func TestInspectJPEGOrPNGDataURLDefersFullDecodeAndEnforcesBudgets(t *testing.T) {
	pngURL := testImageDataURL(t, "image/png")
	payload := strings.TrimPrefix(pngURL, "data:image/png;base64,")
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		t.Fatalf("decode PNG fixture: %v", err)
	}
	truncated := "data:image/png;base64," + base64.StdEncoding.EncodeToString(data[:len(data)-8])
	if _, ok := InspectJPEGOrPNGDataURL(truncated, nil); !ok {
		t.Fatal("metadata inspection should defer truncated pixel-stream rejection to Rust")
	}
	if _, ok := ValidateJPEGOrPNGDataURL(truncated, nil); ok {
		t.Fatal("full Go validation must continue rejecting truncated images for Go-only callers")
	}

	var partsBudget RequestImageBudget
	for index := 0; index < MaxImagePartsPerRequest; index++ {
		if _, ok := InspectJPEGOrPNGDataURL(pngURL, &partsBudget); !ok {
			t.Fatalf("image part %d should fit", index)
		}
	}
	if _, ok := InspectJPEGOrPNGDataURL(pngURL, &partsBudget); ok {
		t.Fatal("metadata inspection exceeded the image-part budget")
	}

	var pixelBudget RequestImageBudget
	maximumArea := pngWithDeclaredDimensions(t, 4096, 4096)
	if _, ok := InspectJPEGOrPNGDataURL(maximumArea, &pixelBudget); !ok {
		t.Fatal("exact aggregate pixel budget should fit")
	}
	if _, ok := InspectJPEGOrPNGDataURL(pngURL, &pixelBudget); ok {
		t.Fatal("metadata inspection exceeded the aggregate pixel budget")
	}
}

func TestRequestImageBudgetBoundsPartsAndAggregatePixels(t *testing.T) {
	var partsBudget RequestImageBudget
	for i := 0; i < MaxImagePartsPerRequest; i++ {
		if !partsBudget.reservePart() {
			t.Fatalf("part %d should fit the request budget", i)
		}
	}
	if partsBudget.reservePart() {
		t.Fatalf("part %d must exceed the request budget", MaxImagePartsPerRequest+1)
	}

	var pixelBudget RequestImageBudget
	if !pixelBudget.reservePixels(MaxImagePixelsPerRequest) {
		t.Fatal("exact aggregate pixel limit should fit")
	}
	if pixelBudget.reservePixels(1) {
		t.Fatal("one pixel beyond aggregate limit must be rejected")
	}

	// The strict validator must charge DecodeConfig metadata to the shared
	// request budget before it performs the full decode.
	nearLimit := RequestImageBudget{pixels: MaxImagePixelsPerRequest - 3}
	if _, ok := ValidateJPEGOrPNGDataURL(testImageDataURL(t, "image/png"), &nearLimit); ok {
		t.Fatal("2x2 image must exceed a request budget with only three pixels left")
	}
}

func TestDecodeBase64(t *testing.T) {
	valid := base64.StdEncoding.EncodeToString([]byte("hello image bytes"))

	cases := []struct {
		name string
		url  string
		want bool
	}{
		{"valid png payload", "data:image/png;base64," + valid, true},
		{"uppercase scheme keeps case-sensitive payload", "DATA:IMAGE/PNG;BASE64," + valid, true},
		{"malformed base64", "data:image/png;base64,!!!!", false},
		{"non-decodable prefix bleed", "DATA:IMAGE/PNG;BASE64," + valid + "===", false},
		{"not an image uri", "https://example.com/cat.png", false},
		{"empty payload", "data:image/png;base64,", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := DecodeBase64(tc.url)
			if ok != tc.want {
				t.Fatalf("DecodeBase64(%q) ok = %v, want %v", tc.url, ok, tc.want)
			}
			if ok && len(got) == 0 {
				t.Fatalf("DecodeBase64(%q) returned empty bytes on success", tc.url)
			}
		})
	}
}

func TestCanonicalDataURL(t *testing.T) {
	// Scheme/MIME/marker lowercased; mixed-case payload preserved verbatim.
	got, ok := CanonicalDataURL("DATA:IMAGE/PNG;BASE64,AbCdEf")
	if !ok {
		t.Fatal("CanonicalDataURL returned ok=false for a safe upper-case data URI")
	}
	if want := "data:image/png;base64,AbCdEf"; got != want {
		t.Fatalf("CanonicalDataURL = %q, want %q", got, want)
	}

	if _, ok := CanonicalDataURL("https://example.com/cat.png"); ok {
		t.Fatal("CanonicalDataURL must reject non-data-URI inputs")
	}
}
