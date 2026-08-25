package postgres

import (
	"fmt"
	"math"
)

func publicRevision(value int64, field string) (uint64, error) {
	if value <= 0 {
		return 0, fmt.Errorf("%s must be positive: %d", field, value)
	}
	// #nosec G115 -- positivity proves every int64 value is representable as uint64.
	return uint64(value), nil
}

func postgresRevision(value uint64, field string) (int64, error) {
	if value == 0 || value > math.MaxInt64 {
		return 0, fmt.Errorf("%s must fit a positive PostgreSQL BIGINT: %d", field, value)
	}
	// #nosec G115 -- the explicit PostgreSQL BIGINT bound proves this conversion is lossless.
	return int64(value), nil
}
