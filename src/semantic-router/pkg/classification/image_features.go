package classification

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"strings"
)

const (
	// MaxImageFeatureCandidates caps total decoding work per request.
	MaxImageFeatureCandidates = 4

	maxFeatureImageBytes  = 2 << 20
	maxFeatureImagePixels = 4 << 20
	featureImageSide      = 384
)

type imageFeatures struct {
	blur, edgeDensity float64
}

func aggregateImageFeatures(ctx context.Context, images []string) (imageFeatures, bool) {
	var maxima imageFeatures
	available := false
	for _, url := range images[:min(len(images), MaxImageFeatureCandidates)] {
		if ctx.Err() != nil {
			break
		}
		if features, ok := extractInlineImageFeatures(ctx, url); ok {
			available = true
			maxima.blur = max(maxima.blur, features.blur)
			maxima.edgeDensity = max(maxima.edgeDensity, features.edgeDensity)
		}
	}
	if ctx.Err() != nil {
		return imageFeatures{}, false
	}
	return maxima, available
}

// extractInlineImageFeatures intentionally accepts only bounded PNG and JPEG
// data URLs. Failure is distinct from a valid image scoring zero.
func extractInlineImageFeatures(ctx context.Context, url string) (imageFeatures, bool) {
	header, payload, ok := strings.Cut(url, ",")
	if !ok || (header != "data:image/png;base64" && header != "data:image/jpeg;base64") || len(payload) > base64.StdEncoding.EncodedLen(maxFeatureImageBytes) {
		return imageFeatures{}, false
	}
	if ctx.Err() != nil {
		return imageFeatures{}, false
	}
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil || len(data) > maxFeatureImageBytes {
		return imageFeatures{}, false
	}
	reader := featureImageReader{ctx: ctx, data: *bytes.NewReader(data)}
	cfg, format, err := image.DecodeConfig(&reader)
	if err != nil || header != "data:image/"+format+";base64" || cfg.Width < 3 || cfg.Height < 3 || cfg.Width > maxFeatureImagePixels/cfg.Height {
		return imageFeatures{}, false
	}
	reader.data.Reset(data)
	im, _, err := image.Decode(&reader)
	if err != nil || im.Bounds().Size() != image.Pt(cfg.Width, cfg.Height) {
		return imageFeatures{}, false
	}
	return measureImageFeatures(ctx, im)
}

// The standard decoders do not take a context. Check it as they consume the
// compressed bytes, as well as while reducing the decoded pixels.
type featureImageReader struct {
	ctx  context.Context
	data bytes.Reader
}

func (r *featureImageReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.data.Read(p)
}

func measureImageFeatures(ctx context.Context, im image.Image) (imageFeatures, bool) {
	b := im.Bounds()
	w, h := b.Dx(), b.Dy()
	if side := max(w, h); side > featureImageSide {
		w, h = max(3, w*featureImageSide/side), max(3, h*featureImageSide/side)
	}
	gray, ok := averageFeaturePixels(ctx, im, b, w, h)
	if !ok {
		return imageFeatures{}, false
	}
	features := measureGrayFeatures(gray, w, h)
	if ctx.Err() != nil {
		return imageFeatures{}, false
	}
	return features, true
}

// Box-average grayscale when reducing dimensions to avoid aliasing dense
// patterns into spurious edges. Accumulate a row at a time for cancellation.
func averageFeaturePixels(ctx context.Context, im image.Image, b image.Rectangle, w, h int) ([]float64, bool) {
	gray := make([]float64, w*h)
	grayAt := featureGraySampler(im)
	xBounds := make([]int, w+1)
	for x := range xBounds {
		xBounds[x] = b.Min.X + x*b.Dx()/w
	}
	for y := range h {
		row := gray[y*w : (y+1)*w]
		y0, y1 := b.Min.Y+y*b.Dy()/h, b.Min.Y+(y+1)*b.Dy()/h
		for sy := y0; sy < y1; sy++ {
			if ctx.Err() != nil {
				return nil, false
			}
			for x, sum := range row {
				for sx := xBounds[x]; sx < xBounds[x+1]; sx++ {
					sum += grayAt(sx, sy)
				}
				row[x] = sum
			}
		}
		for x := range row {
			row[x] /= float64((xBounds[x+1] - xBounds[x]) * (y1 - y0))
		}
	}
	return gray, true
}

// Use concrete pixels from the PNG/JPEG decoders without allocating a
// color.Color per pixel. Keep the GrayModel conversion and white background.
func featureGraySampler(im image.Image) func(int, int) float64 {
	switch img := im.(type) {
	case *image.Gray:
		return func(x, y int) float64 { return float64(img.GrayAt(x, y).Y) }
	case *image.RGBA:
		return func(x, y int) float64 { return featureGrayOnWhite(img.RGBAAt(x, y).RGBA()) }
	case *image.YCbCr:
		return func(x, y int) float64 { return featureGrayOnWhite(img.YCbCrAt(x, y).RGBA()) }
	case image.RGBA64Image:
		return func(x, y int) float64 { return featureGrayOnWhite(img.RGBA64At(x, y).RGBA()) }
	default:
		return func(x, y int) float64 { return featureGrayOnWhite(im.At(x, y).RGBA()) }
	}
}

func featureGrayOnWhite(r, g, b, a uint32) float64 {
	// image/color.GrayModel uses the JFIF coefficients on premultiplied RGB.
	gray := (19595*r + 38470*g + 7471*b + 1<<15) >> 24
	return float64(gray + ((0xffff - a) >> 8))
}

func measureGrayFeatures(gray []float64, w, h int) imageFeatures {
	var lapSum, lapSquares, dx, dy float64
	for y := range h {
		for x := range w {
			i := y*w + x
			if x > 0 {
				dx += math.Abs(gray[i] - gray[i-1])
			}
			if y > 0 {
				dy += math.Abs(gray[i] - gray[i-w])
			}
			if x > 0 && x < w-1 && y > 0 && y < h-1 {
				lap := gray[i-1] + gray[i+1] + gray[i-w] + gray[i+w] - 4*gray[i]
				lapSum += lap
				lapSquares += lap * lap
			}
		}
	}
	n := float64((w - 2) * (h - 2))
	mean := lapSum / n
	variance := lapSquares/n - mean*mean
	edge := (dx/float64(h*(w-1)) + dy/float64(w*(h-1))) / (2 * 255)
	return imageFeatures{blur: min(1, max(0, 1-variance/180)), edgeDensity: min(1, edge/0.025)}
}
