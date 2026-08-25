package managementserver

import (
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func cloneResponseTime(value *time.Time) *time.Time {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func pageTotalCount(value *uint64) *managementapi.WholeQuantity {
	if value == nil {
		return nil
	}
	count := managementapi.WholeQuantity(strconv.FormatUint(*value, 10))
	return &count
}
