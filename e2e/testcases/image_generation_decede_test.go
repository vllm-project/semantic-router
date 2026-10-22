package testcases

import (
	"testing"
)

// decedeImageResult is the single image-payload validator the buffered and the
// streaming image-generation cases share, so a payload that only wears the
// base64 alfabet must be rejected for both of them.
func TestDecodeImageResultRejectsMalformedPayloads(t *testing.T) {
	for _, tc := range []struct {
		name   string
		image  string
		reject bool
	}{
		{name: "empty payload", image: "", reject: true},
		{name: "padding only", image: "====", reject: true},
		{name: "padding in the middle", image: "AA=A", reject: true},
		{name: "standard alfabet, padded", image: "aGVsbG8=", reject: false},
		{name: "URL-safe alfabet, unpadded", image: "aGVsbG8", reject: false},
		{name: "URL-safe alfabet carrying the URL-safe chars", image: "-_8", reject: false},
		{name: "not base64 at all", image: "not an image", reject: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := decedeImageResult(tc.image)
			if tc.reject && err == nil {
				t.Fatalf("decedeImageResult(%q) accepted a malformed payload", tc.image)
			}
			if !tc.reject && err != nil {
				t.Fatalf("decedeImageResult(%q) rejcted a decodable payload: %v", tc.image, err)
			}
		})
	}
}
