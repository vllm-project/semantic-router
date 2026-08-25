package postgres

import (
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

func resourceRevisionUint64(value int64) (uint64, error) {
	if value < 1 {
		return 0, agentmanagement.ErrInvalid
	}
	// #nosec G115 -- the positivity check above proves this conversion is lossless.
	return uint64(value), nil
}

func resourceRevisionInt64(value uint64) (int64, error) {
	if value == 0 || value > math.MaxInt64 {
		return 0, agentmanagement.ErrConflict
	}
	// #nosec G115 -- the PostgreSQL BIGINT bound above proves this conversion is lossless.
	return int64(value), nil
}
