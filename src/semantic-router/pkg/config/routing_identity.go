package config

import (
	"crypto/sha256"
	"fmt"
	"strconv"
	"strings"
)

const initialRoutingResourceRevision int64 = 1

// stableRoutingResourceID deterministically assigns an immutable authoring ID
// when a portable standalone manifest omits one. Managed imports persist this
// value before publication; the data plane never resolves mutable names.
func stableRoutingResourceID(prefix string, parts ...string) string {
	var payload []byte
	for _, part := range parts {
		normalized := strings.TrimSpace(part)
		payload = strconv.AppendInt(payload, int64(len(normalized)), 10)
		payload = append(payload, ':')
		payload = append(payload, normalized...)
		payload = append(payload, ';')
	}
	digest := sha256.Sum256(payload)
	return fmt.Sprintf("%s_%x", prefix, digest[:10])
}

// DeterministicRoutingResourceID assigns a stable identifier when an imported
// authoring format omits one. Runtime references always use the resulting ID;
// names never become an alternate lookup contract.
func DeterministicRoutingResourceID(prefix string, parts ...string) string {
	return stableRoutingResourceID(prefix, parts...)
}
