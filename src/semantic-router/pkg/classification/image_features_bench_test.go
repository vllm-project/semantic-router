package classification

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"testing"
)

// BenchmarkImageFeaturesAtLimit measures decoding and scoring four images at
// the pixel cap. Fixture generation and request JSON parsing are not timed.
func BenchmarkImageFeaturesAtLimit(b *testing.B) {
	for _, fixture := range []struct {
		format string
		encode func(io.Writer, image.Image) error
	}{
		{"png", png.Encode},
		{"jpeg", func(w io.Writer, im image.Image) error { return jpeg.Encode(w, im, nil) }},
	} {
		b.Run(fixture.format, func(b *testing.B) {
			urls := featureBenchmarkURLs(b, fixture.format, fixture.encode)
			b.ReportAllocs()
			for b.Loop() {
				if _, ok := aggregateImageFeatures(b.Context(), urls); !ok {
					b.Fatal("benchmark images were not scored")
				}
			}
		})
	}
}

func featureBenchmarkURLs(b *testing.B, format string, encode func(io.Writer, image.Image) error) []string {
	b.Helper()
	const side = 2048 // side² equals maxFeatureImagePixels.
	im := image.NewRGBA(image.Rect(0, 0, side, side))
	urls := make([]string, MaxImageFeatureCandidates)
	for i := range urls {
		for y := range side {
			for x := range side {
				j := y*im.Stride + 4*x
				im.Pix[j] = uint8((x + y + i*7) & 0xff)
				im.Pix[j+1] = uint8((2*x + y) & 0xff)
				im.Pix[j+2] = uint8(((y/16 + x/16) & 1) * 255)
				im.Pix[j+3] = 255
			}
		}
		var buf bytes.Buffer
		if err := encode(&buf, im); err != nil {
			b.Fatal(err)
		}
		if buf.Len() > maxFeatureImageBytes {
			b.Fatalf("fixture exceeds the byte cap: %d", buf.Len())
		}
		urls[i] = "data:image/" + format + ";base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
	}
	return urls
}
