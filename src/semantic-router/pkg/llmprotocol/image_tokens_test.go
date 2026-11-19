package llmprotocol

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"testing"
)

func encodedImage(t *testing.T, format string, width, height int) string {
	t.Helper()
	picture := image.NewGray(image.Rect(0, 0, width, height))
	var out bytes.Buffer
	var err error
	switch format {
	case "png":
		err = png.Encode(&out, picture)
	case "jpeg":
		err = jpeg.Encode(&out, picture, nil)
	case "gif":
		err = gif.Encode(&out, picture, nil)
	}
	if err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(out.Bytes())
}

// webpHeader is the first 30 bytes of a WebP image of the given chunk kind,
// with its size fields as they are written.
func webpHeader(kind string, size []byte) string {
	header := make([]byte, 30)
	copy(header, "RIFF")
	copy(header[8:], "WEBP")
	copy(header[12:], kind)
	switch kind {
	case "VP8 ":
		copy(header[23:], []byte{0x9D, 0x01, 0x2A})
		copy(header[26:], size)
	case "VP8L":
		header[20] = 0x2F
		copy(header[21:], size)
	case "VP8X":
		copy(header[24:], size)
	}
	return base64.StdEncoding.EncodeToString(header)
}

// An inline image's tokens follow its size, about one per 28 by 28 pixels,
// between the floor and the cap, in every format the estimate reads; an
// image whose size it cannot read keeps the 8K reserve.
func TestInlineImageTokensFollowTheImageSize(t *testing.T) {
	for _, tc := range []struct {
		name, data string
		want       int
	}{
		{"png 100x60", encodedImage(t, "png", 100, 60), 4 * 3},
		{"jpeg 640x480", encodedImage(t, "jpeg", 640, 480), 23 * 18},
		{"gif 28x28", encodedImage(t, "gif", 28, 28), InputImageMinTokens},
		// 1000 and 500 as 14-bit little-endian fields
		{"webp lossy 1000x500", webpHeader("VP8 ", []byte{0xE8, 0x03, 0xF4, 0x01}), 36 * 18},
		// 299 | 299<<14, the sizes less one
		{"webp lossless 300x300", webpHeader("VP8L", []byte{0x2B, 0xC1, 0x4A, 0x00}), 11 * 11},
		// 2999 and 1999, the sizes less one, as 24-bit little-endian fields
		{"webp extended 3000x2000", webpHeader("VP8X", []byte{0xB7, 0x0B, 0x00, 0xCF, 0x07, 0x00}), 108 * 72},
		{"png 4000x4000", encodedImage(t, "png", 4000, 4000), InputImageMaxTokens},
		{"not an image", base64.StdEncoding.EncodeToString([]byte("plain text, not a picture at all")), InputImageTokens},
		{"not base64", "%%%", InputImageTokens},
	} {
		if got := InlineImageTokens(tc.data); got != tc.want {
			t.Errorf("%s: InlineImageTokens = %d, want %d", tc.name, got, tc.want)
		}
	}
	png := encodedImage(t, "png", 100, 60)
	for url, want := range map[string]int{
		"data:image/png;base64," + png:    12,
		"https://example.invalid/a.png":   InputImageTokens,
		"data:text/plain;base64," + png:   InputImageTokens,
		"data:image/png;charset=x," + png: InputImageTokens,
	} {
		if got := DataURLImageTokens(url); got != want {
			t.Errorf("DataURLImageTokens(%.40q) = %d, want %d", url, got, want)
		}
	}
}

// Twenty-five small inline pictures are estimated at a few hundred tokens,
// not at 8K each: the estimate no longer puts them past a 200K window.
func TestSmallInlinePicturesAreNotEstimatedAsLongContext(t *testing.T) {
	data := encodedImage(t, "png", 120, 90)
	content := []Content{{Kind: ContentText, Text: "Which of these icons is a cat?"}}
	for range 25 {
		content = append(content, Content{Kind: ContentImage, MediaType: "image/png", Data: data})
	}
	estimate := EstimateInput(&Request{Messages: []Message{{Role: RoleUser, Content: content}}})
	if estimate.Tokens > 1000 || estimate.Tokens < 25*InputImageMinTokens {
		t.Fatalf("25 pictures of 120x90 are estimated at %d tokens", estimate.Tokens)
	}
}
