package testcases

import (
	"bytes"
	"encoding/base64"
	"image/png"
	"strings"
	"testing"
)

func TestPNGDataURLCarriesItsMarkerInADecodableImage(t *testing.T) {
	for _, marker := range []string{"__SYSTEMONE_CHOICE__ receipt", ""} {
		url := pngDataURL(marker)
		encoded, ok := strings.CutPrefix(url, "data:image/png;base64,")
		if !ok {
			t.Fatalf("pngDataURL(%q) = %q, want a PNG data URL", marker, url)
		}
		raw, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			t.Fatalf("pngDataURL(%q): %v", marker, err)
		}
		if _, err := png.Decode(bytes.NewReader(raw)); err != nil {
			t.Fatalf("pngDataURL(%q) is not a PNG: %v", marker, err)
		}
		hasText := bytes.Contains(raw, []byte("tEXtComment\x00"+marker))
		if hasText != (marker != "") {
			t.Fatalf("pngDataURL(%q) text chunk present = %v", marker, hasText)
		}
	}
}
