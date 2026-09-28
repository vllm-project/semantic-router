package recipe

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"strings"
	"testing"
)

const (
	validOnePixelPNGBase64  = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
	validOnePixelPNGSHA256  = "sha256:431ced6916a2a21a156e38701afe55bbd7f88969fbbfc56d7fe099d47f265460"
	validOnePixelJPEGBase64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////" +
		"////////////////////////////////////////////////2wBDAf//////////////////" +
		"////////////////////////////////////////////////////////////////////wAAR" +
		"CAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAA" +
		"AAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/" +
		"xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/Aaf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA" +
		"/9oACAECAQE/Aaf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Aqf/xAAUEAEAAAAA" +
		"AAAAAAAAAAAAAAAA/9oACAEBAAE/Iaf/2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAA" +
		"AAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//EABQQ" +
		"AQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
)

func TestValidateImageFixturePayloadAcceptsSupportedHeaders(t *testing.T) {
	tests := []struct {
		name      string
		mediaType string
		encoded   string
	}{
		{
			name:      "png",
			mediaType: "image/png",
			encoded:   validOnePixelPNGBase64,
		},
		{
			name:      "gif",
			mediaType: "image/gif",
			encoded:   "R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==",
		},
		{
			name:      "jpeg",
			mediaType: "image/jpeg",
			encoded:   validOnePixelJPEGBase64,
		},
		{
			name:      "webp",
			mediaType: "image/webp",
			encoded:   "UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAUAmJaQAA3AA/v3AgAA=",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			payload, err := base64.StdEncoding.DecodeString(test.encoded)
			if err != nil {
				t.Fatal(err)
			}
			if err := validateImageFixturePayload(payload, test.mediaType, "fixture"); err != nil {
				t.Fatalf("validateImageFixturePayload(): %v", err)
			}
		})
	}
}

func TestProbeManifestRejectsUnsafeImageFixtureHeaders(t *testing.T) {
	validPNG, err := base64.StdEncoding.DecodeString(validOnePixelPNGBase64)
	if err != nil {
		t.Fatal(err)
	}
	staticGIF, err := base64.StdEncoding.DecodeString("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
	if err != nil {
		t.Fatal(err)
	}
	animatedGIF := append([]byte(nil), staticGIF[:len(staticGIF)-1]...)
	animatedGIF = append(animatedGIF, staticGIF[19:len(staticGIF)-1]...)
	animatedGIF = append(animatedGIF, staticGIF[len(staticGIF)-1])
	pixelBomb := pngWithDimensions(t, validPNG, 4097, 4096)
	tests := []struct {
		name      string
		mediaType string
		payload   []byte
		want      string
	}{
		{
			name:      "fake png",
			mediaType: "image/png",
			payload:   []byte("not an encoded image"),
			want:      "valid supported image",
		},
		{
			name:      "declared mime mismatch",
			mediaType: "image/jpeg",
			payload:   validPNG,
			want:      "does not match detected",
		},
		{
			name:      "small compressed pixel bomb",
			mediaType: "image/png",
			payload:   pixelBomb,
			want:      "pixel canvas limit",
		},
		{
			name:      "animated container",
			mediaType: "image/gif",
			payload:   animatedGIF,
			want:      "valid supported image",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			encoded := base64.StdEncoding.EncodeToString(test.payload)
			digest := fmt.Sprintf("sha256:%x", sha256.Sum256(test.payload))
			manifest := validImageFixtureManifest(encoded, digest, "pixel")
			manifest = strings.Replace(manifest, "media_type: image/png", "media_type: "+test.mediaType, 1)
			_, err := decodeProbes([]byte(manifest))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("decodeProbes() error = %v, want %q", err, test.want)
			}
		})
	}
}

func pngWithDimensions(t *testing.T, source []byte, width, height uint32) []byte {
	t.Helper()
	payload := append([]byte(nil), source...)
	if len(payload) < 33 {
		t.Fatalf("PNG fixture has %d bytes", len(payload))
	}
	binary.BigEndian.PutUint32(payload[16:20], width)
	binary.BigEndian.PutUint32(payload[20:24], height)
	binary.BigEndian.PutUint32(payload[29:33], crc32.ChecksumIEEE(payload[12:29]))
	return payload
}
