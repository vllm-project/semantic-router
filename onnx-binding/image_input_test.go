//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package onnx_binding

import (
	"bytes"
	"encoding/base64"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"image/gif"
	"image/png"
	"net/http"
	"net/netip"
	"strings"
	"testing"
)

func TestMultiModalImageInputRejectsUntrustedPayloadsBeforeInference(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		call func() error
	}{
		{
			name: "empty bytes",
			call: func() error {
				_, err := MultiModalEncodeImageFromBytes(nil, 0)
				return err
			},
		},
		{
			name: "oversized bytes",
			call: func() error {
				_, err := MultiModalEncodeImageFromBytes(make([]byte, MaxMultiModalImageEncodedBytes+1), 0)
				return err
			},
		},
		{
			name: "malformed bytes",
			call: func() error {
				_, err := MultiModalEncodeImageFromBytes([]byte("not an image"), 0)
				return err
			},
		},
		{
			name: "unsupported gif",
			call: func() error {
				_, err := MultiModalEncodeImageFromBytes(testGIF(t), 0)
				return err
			},
		},
		{
			name: "oversized dimension",
			call: func() error {
				_, err := MultiModalEncodeImageFromBytes(testPNGWithDeclaredDimensions(t, maxMultiModalImageDimension+1, 1), 0)
				return err
			},
		},
		{
			name: "oversized pixel area",
			call: func() error {
				side := 4097
				_, err := MultiModalEncodeImageFromBytes(testPNGWithDeclaredDimensions(t, side, side), 0)
				return err
			},
		},
		{
			name: "oversized decoded allocation",
			call: func() error {
				_, err := MultiModalEncodeImageFromBytes(testPNGWithDeclaredDimensionsAndBitDepth(t, 4096, 4096, 16), 0)
				return err
			},
		},
		{
			name: "empty base64",
			call: func() error {
				_, err := MultiModalEncodeImageFromBase64("", 0)
				return err
			},
		},
		{
			name: "malformed base64",
			call: func() error {
				_, err := MultiModalEncodeImageFromBase64("not-valid-base64!!!", 0)
				return err
			},
		},
		{
			name: "oversized decoded base64",
			call: func() error {
				payload := base64.StdEncoding.EncodeToString(make([]byte, MaxMultiModalImageEncodedBytes+1))
				_, err := MultiModalEncodeImageFromBase64(payload, 0)
				return err
			},
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := test.call(); !errors.Is(err, ErrInvalidImageInput) {
				t.Fatalf("expected ErrInvalidImageInput, got %v", err)
			}
		})
	}
}

func TestMultiModalImageInputKeepsInferenceFailuresInternal(t *testing.T) {
	t.Parallel()

	_, err := MultiModalEncodeImageFromBytes(testPNG(t), 0)
	if err == nil {
		t.Fatal("expected uninitialized model error")
	}
	if errors.Is(err, ErrInvalidImageInput) {
		t.Fatalf("inference failure must not wrap ErrInvalidImageInput: %v", err)
	}
}

func TestValidateImageGeometryRejectsInvalidDimensions(t *testing.T) {
	t.Parallel()

	for _, dimensions := range [][2]int{{0, 1}, {1, 0}, {-1, 1}, {maxMultiModalImageDimension + 1, 1}, {4097, 4097}} {
		if err := validateImageGeometry(dimensions[0], dimensions[1]); err == nil {
			t.Fatalf("expected %dx%d to be rejected", dimensions[0], dimensions[1])
		}
	}
	if err := validateImageGeometry(4096, 4096); err != nil {
		t.Fatalf("expected boundary geometry to pass: %v", err)
	}
}

func TestImageDestinationAllowsOnlyPublicAddresses(t *testing.T) {
	t.Parallel()

	for _, address := range []string{
		"0.0.0.0",
		"10.0.0.1",
		"100.64.0.1",
		"127.0.0.1",
		"169.254.169.254",
		"172.16.0.1",
		"192.168.0.1",
		"198.18.0.1",
		"::1",
		"fc00::1",
		"fe80::1",
		"2001:db8::1",
	} {
		if isPublicImageDestination(netip.MustParseAddr(address)) {
			t.Fatalf("expected %s to be blocked", address)
		}
	}
	for _, address := range []string{"1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"} {
		if !isPublicImageDestination(netip.MustParseAddr(address)) {
			t.Fatalf("expected %s to be allowed", address)
		}
	}
}

func TestImageDestinationRejectsNonPublicTranslatedIPv4(t *testing.T) {
	t.Parallel()

	for _, ipv4 := range []string{
		"0.0.0.0",
		"10.0.0.1",
		"127.0.0.1",
		"169.254.169.254",
		"198.18.0.1",
	} {
		ipv4 := netip.MustParseAddr(ipv4)
		for _, translated := range []netip.Addr{testONNXRFC6052Address(rfc6052WellKnownTranslationPrefix, ipv4)} {
			if isPublicImageDestination(translated) {
				t.Fatalf("translated non-public destination %s (IPv4 %s) must be blocked", translated, ipv4)
			}
		}
	}
}

func TestImageDestinationAllowsPublicTranslatedIPv4(t *testing.T) {
	t.Parallel()

	ipv4 := netip.MustParseAddr("1.1.1.1")
	for _, translated := range []netip.Addr{testONNXRFC6052Address(rfc6052WellKnownTranslationPrefix, ipv4)} {
		if !isPublicImageDestination(translated) {
			t.Fatalf("translated public destination %s must be allowed", translated)
		}
	}
}

func TestImageDestinationRejectsEntireRFC8215LocalUseBlock(t *testing.T) {
	t.Parallel()

	for _, address := range []string{
		"64:ff9b:1::1",
		"64:ff9b:1::1.1.1.1",
		"64:ff9b:1:abcd:0:5431:101:101",
	} {
		if isPublicImageDestination(netip.MustParseAddr(address)) {
			t.Fatalf("RFC 8215 local-use destination %s must be blocked", address)
		}
	}
}

func TestOnnxImageURLRequiresPublicCredentialFreeHTTPS(t *testing.T) {
	t.Parallel()

	for _, rawURL := range []string{"", "http://example.com/image.png", "https://user:pass@example.com/image.png"} {
		if _, err := MultiModalEncodeImageFromURL(rawURL, 0); !errors.Is(err, ErrInvalidImageInput) {
			t.Fatalf("URL %q must fail as invalid image input, got %v", rawURL, err)
		}
	}
	if _, err := MultiModalEncodeImageFromURL("https://127.0.0.1/private-name.png", 0); !errors.Is(err, ErrInvalidImageInput) {
		t.Fatalf("loopback destination must be blocked as invalid input, got %v", err)
	} else if strings.Contains(err.Error(), "private-name") {
		t.Fatalf("blocked URL must not echo its path, got %v", err)
	}

	client := newOnnxPublicImageClient()
	if transport := client.Transport.(*http.Transport); transport.Proxy != nil || !transport.DisableKeepAlives {
		t.Fatal("image transport must ignore ambient proxies and disable connection reuse")
	}
	if err := client.CheckRedirect(nil, nil); !errors.Is(err, http.ErrUseLastResponse) {
		t.Fatalf("redirects must be rejected, got %v", err)
	}
}

func testPNG(t *testing.T) []byte {
	t.Helper()
	var encoded bytes.Buffer
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.RGBA{R: 1, G: 2, B: 3, A: 255})
	if err := png.Encode(&encoded, img); err != nil {
		t.Fatalf("encode PNG: %v", err)
	}
	return encoded.Bytes()
}

func testONNXRFC6052Address(prefix netip.Prefix, ipv4 netip.Addr) netip.Addr {
	bytes := prefix.Addr().As16()
	v4 := ipv4.As4()
	switch prefix.Bits() {
	case 48:
		bytes[6], bytes[7] = v4[0], v4[1]
		bytes[8] = 0
		bytes[9], bytes[10] = v4[2], v4[3]
	case 96:
		copy(bytes[12:], v4[:])
	default:
		panic("test helper supports only /48 and /96 translation prefixes")
	}
	return netip.AddrFrom16(bytes)
}

func testGIF(t *testing.T) []byte {
	t.Helper()
	var encoded bytes.Buffer
	img := image.NewPaletted(image.Rect(0, 0, 1, 1), color.Palette{color.Black})
	if err := gif.Encode(&encoded, img, nil); err != nil {
		t.Fatalf("encode GIF: %v", err)
	}
	return encoded.Bytes()
}

func testPNGWithDeclaredDimensions(t *testing.T, width, height int) []byte {
	return testPNGWithDeclaredDimensionsAndBitDepth(t, width, height, 8)
}

func testPNGWithDeclaredDimensionsAndBitDepth(t *testing.T, width, height, bitDepth int) []byte {
	t.Helper()
	encoded := append([]byte(nil), testPNG(t)...)
	if len(encoded) < 33 {
		t.Fatal("encoded PNG is too short")
	}
	encoded[16] = byte(uint32(width) >> 24)
	encoded[17] = byte(uint32(width) >> 16)
	encoded[18] = byte(uint32(width) >> 8)
	encoded[19] = byte(width)
	encoded[20] = byte(uint32(height) >> 24)
	encoded[21] = byte(uint32(height) >> 16)
	encoded[22] = byte(uint32(height) >> 8)
	encoded[23] = byte(height)
	encoded[24] = byte(bitDepth)
	checksum := crc32.ChecksumIEEE(encoded[12:29])
	encoded[29] = byte(checksum >> 24)
	encoded[30] = byte(checksum >> 16)
	encoded[31] = byte(checksum >> 8)
	encoded[32] = byte(checksum)
	return encoded
}
