package embedding

import (
	"context"
	"testing"
)

type dimensionCapabilityProvider struct {
	Provider
	dimensions []int
}

func (p dimensionCapabilityProvider) EmbeddingInfo() ModelInfo {
	return ModelInfo{Dimension: p.Dimension(), Dimensions: p.dimensions}
}

func TestResolveDimensionUsesPreparedCapabilities(t *testing.T) {
	for _, size := range []int{384, 768} {
		base, err := NewFuncProvider("test", size, func(context.Context, string) ([]float32, error) {
			t.Fatal("dimension resolution ran inference")
			return nil, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		provider := dimensionCapabilityProvider{base, []int{size}}
		for _, view := range []Provider{provider, WithOptions(provider, Options{})} {
			for _, requested := range []int{0, size} {
				got, err := ResolveDimension(view, requested)
				if err != nil || got != size {
					t.Fatalf("size=%d requested=%d got=%d err=%v", size, requested, got, err)
				}
			}
			for _, requested := range []int{-1, 128, 512} {
				if _, err := ResolveDimension(view, requested); err == nil {
					t.Fatalf("unsupported %d accepted", requested)
				}
			}
		}
		reduced := dimensionCapabilityProvider{base, []int{128, size}}
		if got, err := ResolveDimension(WithOptions(reduced, Options{}), 128); err != nil || got != 128 {
			t.Fatalf("advertised output rejected: %d %v", got, err)
		}
	}
	if _, err := ResolveDimension(nil, 384); err == nil {
		t.Fatal("missing provider accepted")
	}
}
