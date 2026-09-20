//go:build windows || !cgo

package cache

import "context"

// ValidateBackendEmbedding is a no-op for cache backends that are compiled as
// platform stubs. Those builds do not expose the native embedding runtime or
// the search-backed cache implementation, so there is no native inference to
// prewarm at startup.
func ValidateBackendEmbedding(context.Context, LegacyCacheBackend) error {
	return nil
}
