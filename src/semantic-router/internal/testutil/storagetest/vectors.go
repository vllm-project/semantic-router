package storagetest

import (
	"context"
	"crypto/sha256"
	"math"
)

// Vectors supplies deterministic unit vectors to exercise storage/index contracts.
// It does not model language semantics. Aliases declare query/record fixture pairs.
// Real embedding-to-retrieval behavior remains in the Vela live memory suite.
type Vectors struct {
	Size    int
	Aliases map[string]string
}

func (p Vectors) Embed(ctx context.Context, text string) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if alias, ok := p.Aliases[text]; ok {
		text = alias
	}
	sum := sha256.Sum256([]byte(text))
	result := make([]float32, p.Size)
	scale := float32(1 / math.Sqrt(float64(p.Size)))
	for i := range result {
		result[i] = scale
		if sum[(i/8)%len(sum)]&(1<<uint(i%8)) == 0 {
			result[i] = -scale
		}
	}
	return result, nil
}

func (p Vectors) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, text := range texts {
		value, err := p.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		result[i] = value
	}
	return result, nil
}
func (p Vectors) Dimension() int  { return p.Size }
func (p Vectors) Backend() string { return "storage-fixture" }
