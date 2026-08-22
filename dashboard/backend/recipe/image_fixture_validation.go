package recipe

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
)

const (
	maxImageFixtureDimension = 8192
	maxImageFixturePixels    = 16_777_216
)

type imageFixtureHeader struct {
	mediaType string
	width     int
	height    int
}

func validateImageFixturePayload(data []byte, declaredMediaType, label string) error {
	header, ok := inspectImageFixtureHeader(data)
	if !ok {
		return fmt.Errorf("%s.data_base64 is not a valid supported image", label)
	}
	if header.mediaType != declaredMediaType {
		return fmt.Errorf(
			"%s.media_type %q does not match detected %q",
			label,
			declaredMediaType,
			header.mediaType,
		)
	}
	if header.width > maxImageFixtureDimension ||
		header.height > maxImageFixtureDimension ||
		header.width > maxImageFixturePixels/header.height {
		return fmt.Errorf(
			"%s.data_base64 image dimensions %dx%d exceed the %d-pixel side or %d-pixel canvas limit",
			label,
			header.width,
			header.height,
			maxImageFixtureDimension,
			maxImageFixturePixels,
		)
	}
	return nil
}

func inspectImageFixtureHeader(data []byte) (imageFixtureHeader, bool) {
	if len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP" {
		width, height, ok := inspectWebPHeader(data)
		return imageFixtureHeader{mediaType: "image/webp", width: width, height: height}, ok
	}
	configuration, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return imageFixtureHeader{}, false
	}
	mediaType := map[string]string{
		"gif":  "image/gif",
		"jpeg": "image/jpeg",
		"png":  "image/png",
	}[format]
	if mediaType == "" || configuration.Width < 1 || configuration.Height < 1 {
		return imageFixtureHeader{}, false
	}
	switch format {
	case "gif":
		if !validateStaticGIFContainer(data, configuration.Width, configuration.Height) {
			return imageFixtureHeader{}, false
		}
	case "jpeg":
		if !bytes.HasSuffix(data, []byte{0xff, 0xd9}) {
			return imageFixtureHeader{}, false
		}
	case "png":
		if !validateStaticPNGContainer(data) {
			return imageFixtureHeader{}, false
		}
	}
	return imageFixtureHeader{
		mediaType: mediaType,
		width:     configuration.Width,
		height:    configuration.Height,
	}, true
}

func validateStaticPNGContainer(data []byte) bool {
	const signatureBytes = 8
	offset := signatureBytes
	sawImageData := false
	for offset < len(data) {
		if offset+12 > len(data) {
			return false
		}
		chunkLength, ok := boundedBigEndianUint32(data[offset:offset+4], len(data)-offset-12)
		if !ok {
			return false
		}
		payloadEnd := offset + 8 + chunkLength
		chunkEnd := payloadEnd + 4
		chunkType := string(data[offset+4 : offset+8])
		expectedCRC := binary.BigEndian.Uint32(data[payloadEnd:chunkEnd])
		if crc32.ChecksumIEEE(data[offset+4:payloadEnd]) != expectedCRC {
			return false
		}
		if (offset == signatureBytes && chunkType != "IHDR") || (offset != signatureBytes && chunkType == "IHDR") {
			return false
		}
		switch chunkType {
		case "acTL":
			return false
		case "IDAT":
			sawImageData = true
		case "IEND":
			return chunkLength == 0 && sawImageData && chunkEnd == len(data)
		}
		offset = chunkEnd
	}
	return false
}

func validateStaticGIFContainer(data []byte, width, height int) bool {
	if len(data) < 13 {
		return false
	}
	packed := data[10]
	offset := 13
	if packed&0x80 != 0 {
		offset += 3 * (1 << ((packed & 0x07) + 1))
	}
	if offset > len(data) {
		return false
	}
	imageCount := 0
	for offset < len(data) {
		blockType := data[offset]
		offset++
		switch blockType {
		case 0x3b:
			return imageCount == 1 && offset == len(data)
		case 0x21:
			if offset >= len(data) {
				return false
			}
			offset++
			var ok bool
			offset, ok = skipGIFSubBlocks(data, offset)
			if !ok {
				return false
			}
		case 0x2c:
			if offset+9 > len(data) {
				return false
			}
			left := int(binary.LittleEndian.Uint16(data[offset : offset+2]))
			top := int(binary.LittleEndian.Uint16(data[offset+2 : offset+4]))
			frameWidth := int(binary.LittleEndian.Uint16(data[offset+4 : offset+6]))
			frameHeight := int(binary.LittleEndian.Uint16(data[offset+6 : offset+8]))
			descriptor := data[offset+8]
			offset += 9
			if frameWidth < 1 || frameHeight < 1 || left+frameWidth > width || top+frameHeight > height {
				return false
			}
			if descriptor&0x80 != 0 {
				offset += 3 * (1 << ((descriptor & 0x07) + 1))
			}
			if offset >= len(data) || data[offset] < 2 || data[offset] > 8 {
				return false
			}
			offset++
			var ok bool
			offset, ok = skipGIFSubBlocks(data, offset)
			if !ok {
				return false
			}
			imageCount++
			if imageCount > 1 {
				return false
			}
		default:
			return false
		}
	}
	return false
}

func skipGIFSubBlocks(data []byte, offset int) (int, bool) {
	for offset < len(data) {
		blockSize := int(data[offset])
		offset++
		if blockSize == 0 {
			return offset, true
		}
		offset += blockSize
		if offset > len(data) {
			return 0, false
		}
	}
	return 0, false
}

func inspectWebPHeader(data []byte) (int, int, bool) {
	if len(data) < 20 {
		return 0, 0, false
	}
	riffSize, ok := boundedLittleEndianUint32(data[4:8], len(data)-8)
	if !ok || riffSize+8 != len(data) {
		return 0, 0, false
	}
	var canvasWidth, canvasHeight int
	var encodedWidth, encodedHeight int
	hasCanvas := false
	hasEncodedImage := false
	hasImagePayload := false
	for offset := 12; offset < len(data); {
		chunkType, payload, next, ok := nextWebPChunk(data, offset)
		if !ok {
			return 0, 0, false
		}
		switch chunkType {
		case "VP8X":
			if hasCanvas || len(payload) != 10 || payload[0]&0xc1 != 0 ||
				!bytes.Equal(payload[1:4], []byte{0, 0, 0}) {
				return 0, 0, false
			}
			if payload[0]&0x02 != 0 {
				return 0, 0, false
			}
			hasCanvas = true
			canvasWidth = 1 + littleEndianUint24(payload[4:7])
			canvasHeight = 1 + littleEndianUint24(payload[7:10])
		case "VP8 ":
			if hasEncodedImage {
				return 0, 0, false
			}
			encodedWidth, encodedHeight, ok = inspectVP8Payload(payload)
			if !ok {
				return 0, 0, false
			}
			hasEncodedImage = true
			hasImagePayload = true
		case "VP8L":
			if hasEncodedImage {
				return 0, 0, false
			}
			encodedWidth, encodedHeight, ok = inspectVP8LPayload(payload)
			if !ok {
				return 0, 0, false
			}
			hasEncodedImage = true
			hasImagePayload = true
		case "ANIM", "ANMF":
			return 0, 0, false
		}
		offset = next
	}
	if !hasImagePayload {
		return 0, 0, false
	}
	if !hasCanvas {
		return encodedWidth, encodedHeight, encodedWidth > 0 && encodedHeight > 0
	}
	if hasEncodedImage && (encodedWidth > canvasWidth || encodedHeight > canvasHeight) {
		return 0, 0, false
	}
	return canvasWidth, canvasHeight, canvasWidth > 0 && canvasHeight > 0
}

func nextWebPChunk(data []byte, offset int) (string, []byte, int, bool) {
	if offset < 0 || offset+8 > len(data) {
		return "", nil, 0, false
	}
	chunkSize, ok := boundedLittleEndianUint32(data[offset+4:offset+8], len(data)-offset-8)
	if !ok {
		return "", nil, 0, false
	}
	payloadStart := offset + 8
	payloadEnd := payloadStart + chunkSize
	paddedEnd := payloadEnd
	if chunkSize%2 != 0 {
		paddedEnd++
	}
	if paddedEnd > len(data) {
		return "", nil, 0, false
	}
	return string(data[offset : offset+4]), data[payloadStart:payloadEnd], paddedEnd, true
}

func inspectVP8Payload(payload []byte) (int, int, bool) {
	if len(payload) < 10 || payload[0]&1 != 0 || !bytes.Equal(payload[3:6], []byte{0x9d, 0x01, 0x2a}) {
		return 0, 0, false
	}
	width := int(binary.LittleEndian.Uint16(payload[6:8]) & 0x3fff)
	height := int(binary.LittleEndian.Uint16(payload[8:10]) & 0x3fff)
	return width, height, width > 0 && height > 0
}

func inspectVP8LPayload(payload []byte) (int, int, bool) {
	if len(payload) < 5 || payload[0] != 0x2f {
		return 0, 0, false
	}
	bits := binary.LittleEndian.Uint32(payload[1:5])
	if bits>>29 != 0 {
		return 0, 0, false
	}
	return int(bits&0x3fff) + 1, int((bits>>14)&0x3fff) + 1, true
}

func littleEndianUint24(data []byte) int {
	return int(data[0]) | int(data[1])<<8 | int(data[2])<<16
}

func boundedLittleEndianUint32(data []byte, maxValue int) (int, bool) {
	if len(data) < 4 || data[3] != 0 {
		return 0, false
	}
	value := int(data[0]) | int(data[1])<<8 | int(data[2])<<16
	return value, value <= maxValue
}

func boundedBigEndianUint32(data []byte, maxValue int) (int, bool) {
	if len(data) < 4 || data[0] != 0 {
		return 0, false
	}
	value := int(data[1])<<16 | int(data[2])<<8 | int(data[3])
	return value, value <= maxValue
}
