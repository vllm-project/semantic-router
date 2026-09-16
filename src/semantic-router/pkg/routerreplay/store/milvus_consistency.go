package store

import (
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// milvusConsistencyLevel converts the user-facing setting to the SDK enum.
// The replay store defaults to Session so read-modify-write updates are
// read-your-writes even when the setting is omitted.
func milvusConsistencyLevel(value string) (entity.ConsistencyLevel, bool) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "":
		return entity.ClSession, true
	case "strong":
		return entity.ClStrong, true
	case "session":
		return entity.ClSession, true
	case "bounded":
		return entity.ClBounded, true
	case "eventually":
		return entity.ClEventually, true
	default:
		return entity.ClSession, false
	}
}
