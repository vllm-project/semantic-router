package classification

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"
)

func encodeFeatureTestImage(t *testing.T, pixel func(uint8) uint8) string {
	t.Helper()
	im := image.NewGray(image.Rect(0, 0, 256, 256))
	for y := range 256 {
		for x := range 256 {
			im.SetGray(x, y, color.Gray{Y: pixel(uint8(x & 0xff))})
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
		pixel func(uint8) uint8
		want  imageFeatures
	}{
		{"sharp simple", func(x uint8) uint8 { return x / 128 * 255 }, imageFeatures{edgeDensity: 1.0 / 255 / 2 / 0.025}},
		{"sharp dense", func(x uint8) uint8 { return x % 2 * 255 }, imageFeatures{edgeDensity: 1}},
		{"smooth", func(x uint8) uint8 { return x }, imageFeatures{blur: 1, edgeDensity: 1.0 / 255 / 2 / 0.025}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := extractInlineImageFeatures(t.Context(), encodeFeatureTestImage(t, tc.pixel))
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
		if _, ok := extractInlineImageFeatures(t.Context(), url); ok {
			t.Fatal("unexpected score for unsupported or unusable input")
		}
	}
}

func TestInlineImageFeaturesKeepGrayscaleAndAlphaSemantics(t *testing.T) {
	pngImage := image.NewNRGBA(image.Rect(4, 7, 392, 11))
	pngImage.SetNRGBA(4, 7, color.NRGBA{R: 255, B: 255})
	pngImage.SetNRGBA(5, 7, color.NRGBA{R: 100, G: 80, B: 240, A: 127})
	pngImage.SetNRGBA(6, 7, color.NRGBA{R: 120, G: 180, B: 240, A: 255})
	jpegImage := image.NewYCbCr(pngImage.Bounds(), image.YCbCrSubsampleRatio420)
	jpegImage.Y[0], jpegImage.Cb[0], jpegImage.Cr[0] = 21, 144, 245
	for _, im := range []image.Image{pngImage, jpegImage} {
		bounds := im.Bounds()
		gray, ok := averageFeaturePixels(t.Context(), im, bounds, 194, 4)
		if !ok {
			t.Fatal("unexpected cancellation")
		}
		for y := range 4 {
			for x := range 194 {
				p := im.At(bounds.Min.X+2*x, bounds.Min.Y+y)
				q := im.At(bounds.Min.X+2*x+1, bounds.Min.Y+y)
				want := (referenceGrayOnWhite(p) + referenceGrayOnWhite(q)) / 2
				if got := gray[y*194+x]; got != want {
					t.Fatalf("%T box %d,%d: gray = %v, want %v", im, x, y, got, want)
				}
			}
		}
	}
}

func referenceGrayOnWhite(p color.Color) float64 {
	_, _, _, alpha := p.RGBA()
	return float64(color.GrayModel.Convert(p).(color.Gray).Y) + float64((0xffff-alpha)>>8)
}

func TestInlineImageFeaturesStopDuringSampling(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	im := &cancelingFeatureImage{Image: image.NewGray(image.Rect(0, 0, 16, 16)), cancel: cancel}
	if features, ok := measureImageFeatures(ctx, im); ok || features != (imageFeatures{}) || im.reads != 16 {
		t.Fatalf("canceled after the first row: features=%+v available=%v pixels=%d", features, ok, im.reads)
	}
	reader := &featureImageReader{ctx: ctx, data: *bytes.NewReader([]byte("image"))}
	if n, err := reader.Read(make([]byte, 1)); n != 0 || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled decoder read = %d, %v", n, err)
	}
}

type cancelingFeatureImage struct {
	image.Image
	cancel context.CancelFunc
	reads  int
}

func (im *cancelingFeatureImage) At(x, y int) color.Color {
	im.reads++
	if im.reads == 1 {
		im.cancel()
	}
	return im.Image.At(x, y)
}
