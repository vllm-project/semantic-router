package llmprotocol

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"strings"
)

// An inline image's tokens are estimated from its size. The vision encoders
// of open models read about one token per 28 by 28 pixels: 14-pixel patches
// merged two by two, or 448-pixel tiles of 256 tokens. Most resize an image
// into a range of sizes, which the floor and the cap stand for. An image
// whose size the Router cannot read (a link, a file ID, a format other than
// PNG, JPEG, GIF and WebP, or a header past its first imageHeaderBytes)
// keeps the reserve of InputImageTokens.
const (
	InputImageTokenPixels = 28
	InputImageMinTokens   = 4
	InputImageMaxTokens   = 16 * 1024
	imageHeaderBytes      = 256 << 10
)

// InlineImageTokens estimates the tokens of an image sent inline as base64
// data, or InputImageTokens when its size cannot be read.
func InlineImageTokens(data string) int {
	width, height, ok := inlineImageSize(data)
	if !ok {
		return InputImageTokens
	}
	side := func(pixels int) int { return (pixels + InputImageTokenPixels - 1) / InputImageTokenPixels }
	tokens := saturatedMultiply(side(width), side(height))
	return min(max(tokens, InputImageMinTokens), InputImageMaxTokens)
}

// DataURLImageTokens is InlineImageTokens of a base64 data URL, or
// InputImageTokens for any other URL.
func DataURLImageTokens(url string) int {
	header, data, found := strings.Cut(strings.TrimSpace(url), ",")
	if !found || !strings.HasPrefix(strings.ToLower(header), "data:image/") || !strings.HasSuffix(strings.ToLower(header), ";base64") {
		return InputImageTokens
	}
	return InlineImageTokens(data)
}

// imageTokens estimates the tokens of one image content.
func imageTokens(content Content) int {
	if content.Data != "" {
		return InlineImageTokens(content.Data)
	}
	return DataURLImageTokens(content.URL)
}

// inlineImageSize reads the width and height in the header of a base64 PNG,
// JPEG, GIF or WebP image, decoding no more of it than the header needs.
func inlineImageSize(data string) (int, int, bool) {
	header := bufio.NewReaderSize(io.LimitReader(base64.NewDecoder(base64.StdEncoding, strings.NewReader(data)), imageHeaderBytes), 64)
	magic, err := header.Peek(30)
	if err != nil && len(magic) < 12 {
		return 0, 0, false
	}
	var decodeConfig func(io.Reader) (image.Config, error)
	switch {
	case bytes.HasPrefix(magic, []byte("\x89PNG\r\n\x1a\n")):
		decodeConfig = png.DecodeConfig
	case bytes.HasPrefix(magic, []byte{0xFF, 0xD8, 0xFF}):
		decodeConfig = jpeg.DecodeConfig
	case bytes.HasPrefix(magic, []byte("GIF8")):
		decodeConfig = gif.DecodeConfig
	case bytes.HasPrefix(magic, []byte("RIFF")) && bytes.Equal(magic[8:12], []byte("WEBP")):
		return webpSize(magic)
	default:
		return 0, 0, false
	}
	config, err := decodeConfig(header)
	if err != nil || config.Width <= 0 || config.Height <= 0 {
		return 0, 0, false
	}
	return config.Width, config.Height, true
}

// webpSize reads a WebP image's size from its first 30 bytes: the frame
// header of a lossy (VP8) or lossless (VP8L) image, or the canvas of an
// extended one (VP8X).
func webpSize(header []byte) (int, int, bool) {
	if len(header) < 30 {
		return 0, 0, false
	}
	var width, height int
	switch string(header[12:16]) {
	case "VP8 ":
		if !bytes.Equal(header[23:26], []byte{0x9D, 0x01, 0x2A}) {
			return 0, 0, false
		}
		width = int(binary.LittleEndian.Uint16(header[26:28]) & 0x3FFF)
		height = int(binary.LittleEndian.Uint16(header[28:30]) & 0x3FFF)
	case "VP8L":
		if header[20] != 0x2F {
			return 0, 0, false
		}
		bits := binary.LittleEndian.Uint32(header[21:25])
		width, height = int(bits&0x3FFF)+1, int(bits>>14&0x3FFF)+1
	case "VP8X":
		width = int(uint32(header[24])|uint32(header[25])<<8|uint32(header[26])<<16) + 1
		height = int(uint32(header[27])|uint32(header[28])<<8|uint32(header[29])<<16) + 1
	default:
		return 0, 0, false
	}
	return width, height, width > 0 && height > 0
}
