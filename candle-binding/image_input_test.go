//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package candle_binding

import (
	"errors"
	"net/http"
	"net/netip"
	"strings"
	"testing"
)

func TestCandleDirectImageTensorFailsBeforeFFI(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		pixels        []float32
		height, width int
	}{
		{name: "negative height", pixels: []float32{1}, height: -1, width: 1},
		{name: "zero width", pixels: []float32{1}, height: 1, width: 0},
		{name: "oversized side", pixels: []float32{1}, height: maxCandleImageDimension + 1, width: 1},
		{name: "oversized area", pixels: []float32{1}, height: 4097, width: 4097},
		{name: "length mismatch", pixels: []float32{1}, height: 1, width: 1},
	}
	for _, test := range tests {
		if _, err := MultiModalEncodeImage(test.pixels, test.height, test.width, 0); err == nil {
			t.Fatalf("%s must fail before inference", test.name)
		}
	}
}

func TestCandleDirectAudioTensorFailsBeforeFFI(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name              string
		mel               []float32
		nMels, timeFrames int
	}{
		{name: "negative dimensions", mel: []float32{1}, nMels: -1, timeFrames: -1},
		{name: "short backing array", mel: []float32{1}, nMels: 80, timeFrames: 3000},
		{name: "long backing array", mel: []float32{1, 2}, nMels: 1, timeFrames: 1},
		{name: "C int narrowing", mel: []float32{1}, nMels: 1 << 31, timeFrames: 1},
	}
	for _, test := range tests {
		if _, err := MultiModalEncodeAudio(test.mel, test.nMels, test.timeFrames, 0); err == nil {
			t.Fatalf("%s must fail before inference", test.name)
		}
	}
}

func TestCandleImageURLRequiresPublicCredentialFreeHTTPS(t *testing.T) {
	t.Parallel()

	for _, rawURL := range []string{"", "http://example.com/image.png", "https://user:pass@example.com/image.png"} {
		if _, err := MultiModalEncodeImageFromURL(rawURL, 0); !errors.Is(err, ErrInvalidImageInput) {
			t.Fatalf("URL %q must fail as invalid image input, got %v", rawURL, err)
		}
	}
	if _, err := MultiModalEncodeImageFromURL("https://127.0.0.1/private-name.png", 0); err == nil {
		t.Fatal("loopback destination must be blocked")
	} else if !errors.Is(err, ErrInvalidImageInput) || strings.Contains(err.Error(), "private-name") {
		t.Fatalf("blocked URL must use the typed error without echoing its path, got %v", err)
	}

	client := newCandlePublicImageClient()
	if transport := client.Transport.(*http.Transport); transport.Proxy != nil || !transport.DisableKeepAlives {
		t.Fatal("image transport must ignore ambient proxies and disable connection reuse")
	}
	if err := client.CheckRedirect(nil, nil); !errors.Is(err, http.ErrUseLastResponse) {
		t.Fatalf("redirects must be rejected, got %v", err)
	}
}

func TestCandleImageDestinationAllowsOnlyPublicAddresses(t *testing.T) {
	t.Parallel()

	for _, address := range []string{
		"0.0.0.0", "10.0.0.1", "100.64.0.1", "127.0.0.1", "169.254.169.254",
		"172.16.0.1", "192.168.0.1", "198.18.0.1", "::1", "fc00::1",
		"fe80::1", "2001:db8::1",
	} {
		if isPublicCandleImageDestination(netip.MustParseAddr(address)) {
			t.Fatalf("expected %s to be blocked", address)
		}
	}
	for _, address := range []string{"1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"} {
		if !isPublicCandleImageDestination(netip.MustParseAddr(address)) {
			t.Fatalf("expected %s to be allowed", address)
		}
	}
}
