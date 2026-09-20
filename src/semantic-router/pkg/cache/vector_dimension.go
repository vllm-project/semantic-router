package cache

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// VectorDimensionMismatchError identifies a persistent cache resource whose
// vector width does not match the prepared embedding representation.
type VectorDimensionMismatchError struct {
	Backend           string
	CollectionName    string
	StoredDimension   int
	ExpectedDimension int
}

func (e *VectorDimensionMismatchError) Error() string {
	return fmt.Sprintf("%s cache %s vector dimension mismatch: stored=%d expected=%d",
		e.Backend, e.CollectionName, e.StoredDimension, e.ExpectedDimension)
}

func valkeyMetadataKey(raw any) string {
	switch value := raw.(type) {
	case string:
		return value
	case []byte:
		return string(value)
	default:
		return fmt.Sprint(value)
	}
}

func parseValkeyDimension(raw any) (int, bool) {
	var dimension int64
	switch value := raw.(type) {
	case int:
		dimension = int64(value)
	case int8:
		dimension = int64(value)
	case int16:
		dimension = int64(value)
	case int32:
		dimension = int64(value)
	case int64:
		dimension = value
	case uint:
		if uint64(value) > math.MaxInt64 {
			return 0, false
		}
		dimension = int64(value)
	case uint8:
		dimension = int64(value)
	case uint16:
		dimension = int64(value)
	case uint32:
		dimension = int64(value)
	case uint64:
		if value > uint64(^uint(0)>>1) {
			return 0, false
		}
		dimension = int64(value)
	case float64:
		dimension = int64(value)
	case string:
		parsed, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
		if err != nil {
			return 0, false
		}
		dimension = parsed
	case []byte:
		parsed, err := strconv.ParseInt(strings.TrimSpace(string(value)), 10, 64)
		if err != nil {
			return 0, false
		}
		dimension = parsed
	default:
		return 0, false
	}
	if dimension <= 0 || dimension > int64(^uint(0)>>1) {
		return 0, false
	}
	return int(dimension), true
}
