//go:build windows || !cgo

package candle_binding

import (
	"context"
	"errors"
	"net/netip"
	"testing"
	"time"
)

func TestMockEmbeddingDimensionsMatchNativeFallbacks(t *testing.T) {
	for _, test := range []struct {
		name       string
		targetDim  int
		wantLength int
	}{
		{name: "default", targetDim: 0, wantLength: 768},
		{name: "matryoshka", targetDim: 256, wantLength: 256},
		{name: "qwen full", targetDim: 1024, wantLength: 1024},
		{name: "oversized", targetDim: 2048, wantLength: 1024},
	} {
		t.Run(test.name, func(t *testing.T) {
			embedding, err := GetEmbeddingWithDim("valid", 0.5, 0.5, test.targetDim)
			if err != nil {
				t.Fatal(err)
			}
			if len(embedding) != test.wantLength {
				t.Fatalf("embedding length = %d, want %d", len(embedding), test.wantLength)
			}
		})
	}

	multimodal, err := MultiModalEncodeText("valid", 1024)
	if err != nil {
		t.Fatal(err)
	}
	if len(multimodal.Embedding) != 384 {
		t.Fatalf("multimodal oversized fallback = %d, want 384", len(multimodal.Embedding))
	}
}

func TestMockSimilarityHonorsInputAndTopKContracts(t *testing.T) {
	if _, err := CalculateEmbeddingSimilarityWithOptions(
		"valid",
		"hidden\x00suffix",
		SimilarityOptions{ModelType: "auto"},
	); !errors.Is(err, errEmbeddedNULByte) {
		t.Fatalf("embedded NUL error = %v", err)
	}

	result, err := CalculateSimilarityBatchWithOptions(
		"query",
		[]string{"one", "two", "three"},
		1,
		SimilarityOptions{ModelType: "auto"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Matches) != 1 {
		t.Fatalf("match count = %d, want 1", len(result.Matches))
	}
	if _, err := CalculateSimilarityBatchWithOptions(
		"query", nil, 1, SimilarityOptions{ModelType: "auto"},
	); err == nil {
		t.Fatal("empty candidates unexpectedly succeeded")
	}
}

func TestMockLegacyInputsMatchNativeFailClosedContract(t *testing.T) {
	invalid := "trusted\x00hidden"
	if _, err := TokenizeText(invalid, 512); !errors.Is(err, errEmbeddedNULByte) {
		t.Fatalf("TokenizeText embedded-NUL error = %v", err)
	}
	if _, err := GetEmbedding(invalid, 512); !errors.Is(err, errEmbeddedNULByte) {
		t.Fatalf("GetEmbedding embedded-NUL error = %v", err)
	}
	if got := CalculateSimilarity("valid", invalid, 512); got != -1 {
		t.Fatalf("CalculateSimilarity embedded-NUL result = %v", got)
	}
	if got := FindMostSimilar("valid", []string{invalid}, 512); got.Index != -1 || got.Score != -1 {
		t.Fatalf("FindMostSimilar embedded-NUL result = %#v", got)
	}
}

func TestMockAudioValidatesTensorShape(t *testing.T) {
	if _, err := MultiModalEncodeAudio([]float32{1}, 80, 3000, 0); err == nil {
		t.Fatal("short audio tensor unexpectedly succeeded")
	}
	result, err := MultiModalEncodeAudio([]float32{1, 2}, 1, 2, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embedding) != 384 {
		t.Fatalf("audio oversized fallback = %d, want 384", len(result.Embedding))
	}
}

func TestMockImageURLRejectsNonPublicLiteralDestinations(t *testing.T) {
	for _, rawURL := range []string{
		"https://127.0.0.1/image.png",
		"https://[::1]/image.png",
		"https://[64:ff9b::7f00:1]/image.png",
	} {
		if _, err := MultiModalEncodeImageFromURL(rawURL, 0); !errors.Is(err, ErrInvalidImageInput) {
			t.Fatalf("URL %q error = %v, want invalid-image rejection", rawURL, err)
		}
	}
}

func TestMockImageURLUsesDiscoveredPref64ForLiteralDestinations(t *testing.T) {
	prefix := netip.MustParsePrefix("2001:4860:1234:5600::/56")
	resolver := candleImageResolverFunc(func(_ context.Context, network, _ string) ([]netip.Addr, error) {
		if network != "ip6" {
			return nil, errors.New("unexpected lookup network")
		}
		return []netip.Addr{
			synthesizeCandleRFC6052Address(prefix, rfc7050IPv4OnlyAddresses[0]),
			synthesizeCandleRFC6052Address(prefix, rfc7050IPv4OnlyAddresses[1]),
		}, nil
	})
	cache := newPref64DiscoveryCache(resolver, time.Minute, time.Second, time.Second)
	previousCache := candleImagePref64Cache
	candleImagePref64Cache = cache
	t.Cleanup(func() { candleImagePref64Cache = previousCache })

	private := synthesizeCandleRFC6052Address(prefix, netip.MustParseAddr("169.254.169.254"))
	if _, err := MultiModalEncodeImageFromURL("https://["+private.String()+"]/image.png", 0); !errors.Is(err, ErrInvalidImageInput) {
		t.Fatalf("NSP metadata literal error = %v, want invalid-image rejection", err)
	}

	public := synthesizeCandleRFC6052Address(prefix, netip.MustParseAddr("1.1.1.1"))
	if _, err := MultiModalEncodeImageFromURL("https://["+public.String()+"]/image.png", 0); err != nil {
		t.Fatalf("NSP public literal unexpectedly rejected: %v", err)
	}
}
