package main

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
)

func TestRecompressPNGKeepsPixels(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 64, 32))
	for y := uint8(0); y < 32; y++ {
		for x := uint8(0); x < 64; x++ {
			img.Set(int(x), int(y), color.NRGBA{R: x * 4, G: y * 8, B: x ^ y, A: 255 - x})
		}
	}
	var original bytes.Buffer
	if err := (&png.Encoder{CompressionLevel: png.NoCompression}).Encode(&original, img); err != nil {
		t.Fatal(err)
	}
	smaller, err := recompressPNG(original.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if len(smaller) >= original.Len() {
		t.Fatalf("re-encoding did not shrink the image: %d >= %d bytes", len(smaller), original.Len())
	}
	decoded, err := png.Decode(bytes.NewReader(smaller))
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := decoded.(*image.NRGBA); !ok || !bytes.Equal(got.Pix, img.Pix) {
		t.Fatalf("re-encoding changed the pixels (%T)", decoded)
	}
	if _, err := recompressPNG([]byte("not a png")); err == nil {
		t.Fatal("a non-PNG payload was accepted")
	}
}
