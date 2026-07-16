package imageurl

import (
	"encoding/base64"
	"testing"
)

func TestIsSafeImageDataURL(t *testing.T) {
	cases := []struct {
		name string
		url  string
		want bool
	}{
		{"png data uri", "data:image/png;base64,iVBORw0KGgo=", true},
		{"jpeg data uri", "data:image/jpeg;base64,/9j/4AAQ", true},
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
