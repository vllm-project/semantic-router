//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package onnx_binding

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net"
	"net/http"
	urlpkg "net/url"
	"strings"
	"time"
)

// ErrInvalidImageInput identifies image data that cannot be decoded safely by
// the multimodal image pipeline. Model, device, tensor, and inference failures
// do not wrap this sentinel.
var ErrInvalidImageInput = errors.New("invalid image input")

// MaxMultiModalImageEncodedBytes is the largest encoded JPEG/PNG payload the
// multimodal image pipeline accepts through any input transport.
const MaxMultiModalImageEncodedBytes = 20 * 1024 * 1024

const (
	maxMultiModalImageDimension = 8192
	maxMultiModalImagePixels    = 16_777_216
	maxMultiModalImageDecode    = 64 * 1024 * 1024
	multiModalImageSize         = 512
)

// MultiModalEncodeImageFromBytes decodes bounded JPEG/PNG bytes, resizes to
// 512x512, and encodes them into a multimodal embedding.
func MultiModalEncodeImageFromBytes(imageBytes []byte, targetDim int) (*MultiModalEmbeddingOutput, error) {
	if err := validateONNXNonNegativeCInt("target dimension", targetDim); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidImageInput, err)
	}
	if len(imageBytes) == 0 {
		return nil, fmt.Errorf("%w: imageBytes cannot be empty", ErrInvalidImageInput)
	}
	if len(imageBytes) > MaxMultiModalImageEncodedBytes {
		return nil, fmt.Errorf("%w: image payload exceeds %d bytes", ErrInvalidImageInput, MaxMultiModalImageEncodedBytes)
	}

	pixelData, err := decodeAndResizeImageOnnx(imageBytes, multiModalImageSize, multiModalImageSize)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidImageInput, err)
	}
	return MultiModalEncodeImage(pixelData, multiModalImageSize, multiModalImageSize, targetDim)
}

// MultiModalEncodeImageFromBase64 decodes a bounded base64 image and encodes it.
func MultiModalEncodeImageFromBase64(base64Str string, targetDim int) (*MultiModalEmbeddingOutput, error) {
	if err := validateONNXNonNegativeCInt("target dimension", targetDim); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidImageInput, err)
	}
	if base64Str == "" {
		return nil, fmt.Errorf("%w: base64Str cannot be empty", ErrInvalidImageInput)
	}
	payload := base64Str
	if idx := strings.Index(base64Str, ";base64,"); idx >= 0 {
		payload = base64Str[idx+len(";base64,"):]
	}

	decoder := base64.NewDecoder(base64.StdEncoding, strings.NewReader(payload))
	data, err := io.ReadAll(io.LimitReader(decoder, MaxMultiModalImageEncodedBytes+1))
	if err != nil {
		return nil, fmt.Errorf("%w: base64 decode failed: %v", ErrInvalidImageInput, err)
	}
	if len(data) > MaxMultiModalImageEncodedBytes {
		return nil, fmt.Errorf("%w: image payload exceeds %d bytes", ErrInvalidImageInput, MaxMultiModalImageEncodedBytes)
	}
	return MultiModalEncodeImageFromBytes(data, targetDim)
}

// MultiModalEncodeImageFromURL downloads a bounded image from a public HTTPS
// destination. DNS results are checked and the connection is pinned to those
// public addresses to prevent private-target access and DNS rebinding.
func MultiModalEncodeImageFromURL(rawURL string, targetDim int) (*MultiModalEmbeddingOutput, error) {
	if err := validateONNXNonNegativeCInt("target dimension", targetDim); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidImageInput, err)
	}
	parsed, err := urlpkg.Parse(rawURL)
	if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil {
		return nil, fmt.Errorf("%w: a credential-free https URL with a host is required", ErrInvalidImageInput)
	}

	client := newOnnxPublicImageClient()
	resp, err := client.Get(parsed.String())
	if err != nil {
		return nil, fmt.Errorf("%w: image download rejected: %v", ErrInvalidImageInput, sanitizedOnnxImageRequestError(err))
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP GET returned status %d", resp.StatusCode)
	}
	if resp.ContentLength > MaxMultiModalImageEncodedBytes {
		return nil, fmt.Errorf("%w: remote image exceeds %d bytes", ErrInvalidImageInput, MaxMultiModalImageEncodedBytes)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, MaxMultiModalImageEncodedBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read body error: %w", err)
	}
	if len(data) > MaxMultiModalImageEncodedBytes {
		return nil, fmt.Errorf("%w: remote image exceeds %d bytes", ErrInvalidImageInput, MaxMultiModalImageEncodedBytes)
	}
	return MultiModalEncodeImageFromBytes(data, targetDim)
}

func newOnnxPublicImageClient() *http.Client {
	return &http.Client{
		Timeout:   30 * time.Second,
		Transport: publicImageTransport(),
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

func sanitizedOnnxImageRequestError(err error) error {
	var requestError *urlpkg.Error
	if errors.As(err, &requestError) {
		return requestError.Err
	}
	return err
}

// decodeAndResizeImageOnnx performs geometry preflight before allocating the
// decoded source image. The standard-library decoders are restricted to JPEG
// and PNG by the explicit format check.
func decodeAndResizeImageOnnx(data []byte, targetW, targetH int) ([]float32, error) {
	if err := validateImageGeometry(targetW, targetH); err != nil {
		return nil, fmt.Errorf("invalid resize target: %w", err)
	}
	img, srcW, srcH, err := decodeImageOnnx(data)
	if err != nil {
		return nil, err
	}
	bounds := img.Bounds()

	targetPixels := targetW * targetH
	pixels := make([]float32, 3*targetPixels)
	for y := 0; y < targetH; y++ {
		srcY := y * srcH / targetH
		for x := 0; x < targetW; x++ {
			srcX := x * srcW / targetW
			r, g, b, _ := img.At(bounds.Min.X+srcX, bounds.Min.Y+srcY).RGBA()
			index := y*targetW + x
			pixels[index] = float32(r) / 65535.0
			pixels[targetPixels+index] = float32(g) / 65535.0
			pixels[2*targetPixels+index] = float32(b) / 65535.0
		}
	}
	return pixels, nil
}

func decodeImageOnnx(data []byte) (image.Image, int, int, error) {
	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("image metadata decode failed: %w", err)
	}
	if format != "jpeg" && format != "png" {
		return nil, 0, 0, fmt.Errorf("unsupported image format %q", format)
	}
	if err := validateImageGeometry(config.Width, config.Height); err != nil {
		return nil, 0, 0, err
	}
	bytesPerPixel, err := decodedImageBytesPerPixel(format, data)
	if err != nil {
		return nil, 0, 0, err
	}
	if uint64(config.Width)*uint64(config.Height)*uint64(bytesPerPixel) > maxMultiModalImageDecode {
		return nil, 0, 0, fmt.Errorf("decoded image exceeds %d byte allocation budget", maxMultiModalImageDecode)
	}

	img, decodedFormat, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("image decode failed: %w", err)
	}
	if decodedFormat != format {
		return nil, 0, 0, errors.New("decoded image format changed after metadata preflight")
	}
	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()
	if srcW != config.Width || srcH != config.Height {
		return nil, 0, 0, errors.New("decoded image dimensions changed after metadata preflight")
	}
	return img, srcW, srcH, nil
}

func validateImageGeometry(width, height int) error {
	if width <= 0 || height <= 0 {
		return fmt.Errorf("image dimensions must be positive, got %dx%d", width, height)
	}
	if width > maxMultiModalImageDimension || height > maxMultiModalImageDimension {
		return fmt.Errorf("image dimensions %dx%d exceed maximum side %d", width, height, maxMultiModalImageDimension)
	}
	if uint64(width)*uint64(height) > maxMultiModalImagePixels {
		return fmt.Errorf("image dimensions %dx%d exceed maximum pixel count %d", width, height, maxMultiModalImagePixels)
	}
	return nil
}

func decodedImageBytesPerPixel(format string, data []byte) (int, error) {
	if format == "jpeg" {
		return 4, nil
	}
	if format != "png" || len(data) < 29 || !bytes.Equal(data[:8], []byte("\x89PNG\r\n\x1a\n")) ||
		!bytes.Equal(data[12:16], []byte("IHDR")) {
		return 0, errors.New("invalid PNG metadata")
	}
	if data[24] == 16 {
		return 8, nil
	}
	return 4, nil
}

func publicImageTransport() *http.Transport {
	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	return &http.Transport{
		Proxy:             nil,
		DisableKeepAlives: true,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(address)
			if err != nil {
				return nil, fmt.Errorf("split image destination: %w", err)
			}
			addresses, err := net.DefaultResolver.LookupNetIP(ctx, "ip", host)
			if err != nil {
				return nil, fmt.Errorf("resolve image destination: %w", err)
			}
			if len(addresses) == 0 {
				return nil, errors.New("image destination resolved to no addresses")
			}
			if !imageDestinationsArePublic(ctx, addresses, onnxImagePref64Cache) {
				return nil, errors.New("image destination could not be verified as public")
			}
			var dialErr error
			for _, resolved := range addresses {
				conn, err := dialer.DialContext(ctx, network, net.JoinHostPort(resolved.String(), port))
				if err == nil {
					return conn, nil
				}
				dialErr = err
			}
			return nil, fmt.Errorf("connect image destination: %w", dialErr)
		},
		ForceAttemptHTTP2: true,
	}
}
