package postgres

import (
	"fmt"
	"math"
)

func positiveUint64(value int64, field string) (uint64, error) {
	if value <= 0 {
		return 0, fmt.Errorf("%s must be positive: %d", field, value)
	}
	// #nosec G115 -- the positivity check above proves this conversion is lossless.
	return uint64(value), nil
}

func postgresInt64(value uint64, field string) (int64, error) {
	if value > math.MaxInt64 {
		return 0, fmt.Errorf("%s exceeds PostgreSQL BIGINT: %d", field, value)
	}
	// #nosec G115 -- the PostgreSQL BIGINT bound above proves this conversion is lossless.
	return int64(value), nil
}

func nonNegativeUint32(value int64, field string) (uint32, error) {
	if value < 0 || value > math.MaxUint32 {
		return 0, fmt.Errorf("%s must fit an unsigned 32-bit integer: %d", field, value)
	}
	// #nosec G115 -- the explicit uint32 bounds above prove this conversion is lossless.
	return uint32(value), nil
}
