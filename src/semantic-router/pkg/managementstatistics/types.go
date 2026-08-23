// Package managementstatistics provides bounded, authorization-scoped
// cardinality snapshots for Management clients. It intentionally excludes
// usage, latency, and cost: the immutable usage ledger owns those facts.
package managementstatistics

import (
	"context"
	"errors"
	"regexp"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const DefaultExpiringWindow = 30 * 24 * time.Hour

var (
	ErrInvalidRequest = errors.New("invalid Management statistics request")
	ErrUnavailable    = errors.New("management statistics are unavailable")
	countPattern      = regexp.MustCompile(`^(0|[1-9][0-9]*)$`)
)

// Count is an exact non-negative base-10 cardinality. Keeping it as text
// preserves the PostgreSQL COUNT value across JSON and JavaScript clients.
type Count string

func (count Count) Valid() bool { return countPattern.MatchString(string(count)) }

// Scopes contains one independently authorized visibility envelope per field
// family. A nil scope means the caller may not learn that field.
type Scopes struct {
	Users          *accesscontrol.ResultScope
	Teams          *accesscontrol.ResultScope
	APIKeys        *accesscontrol.ResultScope
	AccessPolicies *accesscontrol.ResultScope
	RatePolicies   *accesscontrol.ResultScope
}

type Request struct {
	NamespaceID string
	Scopes      Scopes
}

type Query struct {
	NamespaceID    string
	AsOf           time.Time
	ExpiringBefore time.Time
	Scopes         Scopes
}

type Snapshot struct {
	AsOf               time.Time
	ExpiringBefore     time.Time
	Users              *Count
	Teams              *Count
	ActiveAPIKeys      *Count
	ExpiringAPIKeys    *Count
	AccessPolicies     *Count
	ActiveRatePolicies *Count
}

type Repository interface {
	Ready(context.Context) error
	Snapshot(context.Context, Query) (Snapshot, error)
}
