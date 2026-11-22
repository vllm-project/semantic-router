package classification

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"strings"
)

const (
	maxFeatureImageBytes  = 2 << 20
	maxFeatureImagePixels = 4 << 20
	featureImageSide      = 384
)

type imageFeatures struct {
	blur, edgeDensity float64
}

// extractInlineImageFeatures intentionally accepts only bounded PNG and JPEG
// data URLs. Failure is distinct from a valid image scoring zero.
func extractInlineImageFeatures(url string) (imageFeatures, bool) {
	header, payload, ok := strings.Cut(url, ",")
	if !ok || (header != "data:image/png;base64" && header != "data:image/jpeg;base64") || len(payload) > base64.StdEncoding.EncodedLen(maxFeatureImageBytes) {
		return imageFeatures{}, false
	}
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil || len(data) > maxFeatureImageBytes {
		return imageFeatures{}, false
	}
	cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || header != "data:image/"+format+";base64" || cfg.Width < 3 || cfg.Height < 3 || cfg.Width > maxFeatureImagePixels/cfg.Height {
		return imageFeatures{}, false
	}
	im, _, err := image.Decode(bytes.NewReader(data))
	if err != nil || im.Bounds().Size() != image.Pt(cfg.Width, cfg.Height) {
		return imageFeatures{}, false
	}
	return measureImageFeatures(im), true
}

func measureImageFeatures(im image.Image) imageFeatures {
	b := im.Bounds()
	w, h := b.Dx(), b.Dy()
	if side := max(w, h); side > featureImageSide {
		w, h = max(3, w*featureImageSide/side), max(3, h*featureImageSide/side)
	}
	// Box-average grayscale when reducing dimensions to avoid aliasing dense
	// patterns into spurious edges. Composite transparent pixels over white.
	gray := make([]float64, w*h)
	for y := range h {
		for x := range w {
			x0, x1 := x*b.Dx()/w, (x+1)*b.Dx()/w
			y0, y1 := y*b.Dy()/h, (y+1)*b.Dy()/h
			var sum float64
			for sy := y0; sy < y1; sy++ {
				for sx := x0; sx < x1; sx++ {
					p := im.At(b.Min.X+sx, b.Min.Y+sy)
					_, _, _, alpha := p.RGBA()
					sum += float64(color.GrayModel.Convert(p).(color.Gray).Y) + float64((0xffff-alpha)>>8)
				}
			}
			gray[y*w+x] = sum / float64((x1-x0)*(y1-y0))
		}
	}
	return measureGrayFeatures(gray, w, h)
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
	variance := lapSquares/n - math.Pow(lapSum/n, 2)
	edge := (dx/float64(h*(w-1)) + dy/float64(w*(h-1))) / (2 * 255)
	return imageFeatures{blur: min(1, max(0, 1-variance/180)), edgeDensity: min(1, edge/0.025)}
}
