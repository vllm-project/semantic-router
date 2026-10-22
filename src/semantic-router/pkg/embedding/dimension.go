package embedding

import (
	"fmt"
	"math"
	"slices"
)

// ResolveDimension uses the prepared model's actual representation. A requested
// width must be an advertised output, never a caller-side vector slice.
func ResolveDimension(provider Provider, configured int) (int, error) {
	if configured < 0 || configured > math.MaxInt32 {
		return 0, fmt.Errorf("embedding dimension must fit nonnegative int32")
	}
	if provider == nil {
		return 0, fmt.Errorf("embedding provider was not prepared")
	}
	dimension := provider.Dimension()
	if dimension <= 0 {
		return 0, fmt.Errorf("prepared embedding provider has no output dimension")
	}
	if configured == 0 || configured == dimension {
		return dimension, nil
	}
	if described, ok := provider.(Described); ok && slices.Contains(described.EmbeddingInfo().Dimensions, configured) {
		return configured, nil
	}
	return 0, fmt.Errorf("embedding dimension %d is not supported by prepared provider (default %d)", configured, dimension)
}
