package memory

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestQdrantListTotalBoundsCount(t *testing.T) {
	total, err := qdrantListTotal(uint64(math.MaxInt))
	require.NoError(t, err)
	require.Equal(t, math.MaxInt, total)

	_, err = qdrantListTotal(uint64(math.MaxInt) + 1)
	require.Error(t, err)
}
