package candle_binding

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
)

// ErrInvalidImageInput identifies image data that cannot be decoded by the
// multimodal image pipeline. Callers should use errors.Is rather than matching
// error strings. Model, device, tensor, and inference failures do not wrap this
// sentinel.
var ErrInvalidImageInput = errors.New("invalid image input")

// MaxMultiModalImageEncodedBytes is the largest encoded JPEG/PNG payload the
// multimodal image pipeline will accept, regardless of whether bytes arrive
// directly, through base64, from disk, or from a trusted URL.
const MaxMultiModalImageEncodedBytes = 20 * 1024 * 1024

const (
	maxCandleImageDimension = 8192
	maxCandleImagePixels    = 16_777_216
)

func validateCandleImageTensor(pixelData []float32, width, height int) error {
	if len(pixelData) == 0 {
		return errors.New("pixelData cannot be empty")
	}
	if width <= 0 || height <= 0 {
		return fmt.Errorf("image dimensions must be positive, got %d×%d", width, height)
	}
	if width > maxCandleImageDimension || height > maxCandleImageDimension {
		return fmt.Errorf("image dimensions %d×%d exceed maximum side %d", width, height, maxCandleImageDimension)
	}
	if uint64(width)*uint64(height) > maxCandleImagePixels {
		return fmt.Errorf("image dimensions %d×%d exceed maximum pixel count %d", width, height, maxCandleImagePixels)
	}
	expected := 3 * width * height
	if len(pixelData) != expected {
		return fmt.Errorf("pixelData length %d does not match 3×%d×%d", len(pixelData), height, width)
	}
	return nil
}

func validateCandleEncodedImage(imageBytes []byte) error {
	if len(imageBytes) == 0 {
		return fmt.Errorf("%w: imageBytes cannot be empty", ErrInvalidImageInput)
	}
	if len(imageBytes) > MaxMultiModalImageEncodedBytes {
		return fmt.Errorf("%w: image payload exceeds %d bytes", ErrInvalidImageInput, MaxMultiModalImageEncodedBytes)
	}
	config, format, err := image.DecodeConfig(bytes.NewReader(imageBytes))
	if err != nil || (format != "jpeg" && format != "png") {
		return fmt.Errorf("%w: image bytes could not be decoded", ErrInvalidImageInput)
	}
	if config.Width <= 0 || config.Height <= 0 ||
		config.Width > maxCandleImageDimension || config.Height > maxCandleImageDimension ||
		uint64(config.Width)*uint64(config.Height) > maxCandleImagePixels {
		return fmt.Errorf("%w: image geometry exceeds the decode budget", ErrInvalidImageInput)
	}
	return nil
}

func multiModalImageStatusError(status int) error {
	if status == -2 {
		return fmt.Errorf("%w: image bytes could not be decoded", ErrInvalidImageInput)
	}
	return fmt.Errorf("multi-modal image encoding from bytes failed (status %d)", status)
}

// MultiModalEmbeddingOutput represents the result of a multi-modal embedding.
// It is shared by the native and non-CGO implementations so both build modes
// expose exactly the same public contract.
type MultiModalEmbeddingOutput struct {
	Embedding        []float32 // The embedding vector (384-dim by default)
	Modality         string    // "text", "image", or "audio"
	ProcessingTimeMs float32   // Processing time in milliseconds
}

// ModalityResult represents the output of modality routing classification.
type ModalityResult struct {
	Modality   string  // "AR", "DIFFUSION", or "BOTH"
	ClassID    int     // 0=AR, 1=DIFFUSION, 2=BOTH
	Confidence float32 // Confidence score (0.0-1.0)
}
