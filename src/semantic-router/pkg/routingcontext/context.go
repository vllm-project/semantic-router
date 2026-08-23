// Package routingcontext carries the immutable managed routing generation
// selected for one request. It contains no credential, policy document, or
// mutable routing state.
package routingcontext

import (
	"context"
	"errors"
	"strings"
)

// Generation is the complete request pin for one namespace Router generation.
// Every field is required so internal subrequests can never fall back to a
// default namespace or silently advance to a newer publication.
type Generation struct {
	NamespaceID      string
	QuotaPartition   string
	PublicationID    string
	RuntimeEpoch     uint64
	SnapshotRevision int64
	RoutingDigest    string
}

// Validate rejects partial or malformed generation pins.
func (generation Generation) Validate() error {
	if strings.TrimSpace(generation.NamespaceID) == "" ||
		strings.TrimSpace(generation.QuotaPartition) == "" ||
		strings.TrimSpace(generation.PublicationID) == "" ||
		generation.RuntimeEpoch == 0 || generation.SnapshotRevision <= 0 ||
		!validDigest(generation.RoutingDigest) {
		return errors.New("managed routing generation is incomplete")
	}
	return nil
}

type generationContextKey struct{}

// WithGeneration binds a validated immutable generation to ctx.
func WithGeneration(ctx context.Context, generation Generation) (context.Context, error) {
	if ctx == nil {
		return nil, errors.New("routing context is required")
	}
	if err := generation.Validate(); err != nil {
		return nil, err
	}
	return context.WithValue(ctx, generationContextKey{}, generation), nil
}

// GenerationFrom returns the validated generation bound to ctx.
func GenerationFrom(ctx context.Context) (Generation, bool) {
	if ctx == nil {
		return Generation{}, false
	}
	generation, ok := ctx.Value(generationContextKey{}).(Generation)
	if !ok || generation.Validate() != nil {
		return Generation{}, false
	}
	return generation, true
}

func validDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, character := range value {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}
