package postgres

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func databaseRevision(value int64, field string) (accesscontrol.Revision, error) {
	if value <= 0 {
		return 0, fmt.Errorf("%s must be positive: %d", field, value)
	}
	// #nosec G115 -- the positivity check above proves this conversion is lossless.
	return accesscontrol.Revision(value), nil
}

func databaseRevisionUint64(value int64, field string) (uint64, error) {
	if value <= 0 {
		return 0, fmt.Errorf("%s must be positive: %d", field, value)
	}
	// #nosec G115 -- the positivity check above proves this conversion is lossless.
	return uint64(value), nil
}
