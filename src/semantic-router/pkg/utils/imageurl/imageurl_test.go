package imageurl

import "testing"

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
