package classification

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"
)

func encodeFeatureTestImage(t *testing.T, pixel func(int) uint8) string {
	t.Helper()
	im := image.NewGray(image.Rect(0, 0, 256, 256))
	for y := range 256 {
		for x := range 256 {
			im.SetGray(x, y, color.Gray{Y: pixel(x)})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, im); err != nil {
		t.Fatal(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

func TestInlineImageFeatures(t *testing.T) {
	for _, tc := range []struct {
		name  string
		pixel func(int) uint8
		want  imageFeatures
	}{
		{"sharp simple", func(x int) uint8 { return uint8(x / 128 * 255) }, imageFeatures{edgeDensity: 1.0 / 255 / 2 / 0.025}},
		{"sharp dense", func(x int) uint8 { return uint8(x % 2 * 255) }, imageFeatures{edgeDensity: 1}},
		{"smooth", func(x int) uint8 { return uint8(x) }, imageFeatures{blur: 1, edgeDensity: 1.0 / 255 / 2 / 0.025}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := extractInlineImageFeatures(encodeFeatureTestImage(t, tc.pixel))
			if !ok || got != tc.want {
				t.Fatalf("got %+v, available=%v; want %+v", got, ok, tc.want)
			}
		})
	}
}

func TestInlineImageFeaturesUnavailable(t *testing.T) {
	for _, url := range []string{
		"",
		"https://example.com/image.png",
		"data:image/png;base64,not-valid",
		"data:image/png;base64,YWJj",
		"data:image/png;base64," + strings.Repeat("A", base64.StdEncoding.EncodedLen(maxFeatureImageBytes)+1),
	} {
		if _, ok := extractInlineImageFeatures(url); ok {
			t.Fatal("unexpected score for unsupported or unusable input")
		}
	}
}
