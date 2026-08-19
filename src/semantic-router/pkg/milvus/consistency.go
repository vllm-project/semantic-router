package milvus

import (
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// ConsistencyLevelNames lists the accepted consistency level names, for use
// in configuration warnings.
const ConsistencyLevelNames = "Strong, Session, Bounded, Eventually"

// ParseConsistencyLevel maps a configured consistency level name to the SDK
// constant. Names are matched case-insensitively with surrounding whitespace
// trimmed. ok is false when the name is empty or unrecognized; the returned
// level is the type's zero value then and must be ignored — each caller
// applies its own fallback policy (e.g. the semantic cache leaves the SDK
// default in effect, the router replay store defaults to Session).
func ParseConsistencyLevel(name string) (entity.ConsistencyLevel, bool) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "strong":
		return entity.ClStrong, true
	case "session":
		return entity.ClSession, true
	case "bounded":
		return entity.ClBounded, true
	case "eventually":
		return entity.ClEventually, true
	default:
		return entity.ClStrong, false
	}
}
