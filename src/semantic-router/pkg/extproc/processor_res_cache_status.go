package extproc

// nonSuccessStatusReason is the canonical skip-reason recorded when a cache
// write is skipped because the upstream returned a non-2xx HTTP status. It
// shares the tracing.AttrCacheWriteSkippedReason key with the retention and
// personalized-context skip causes so operators can query a single attribute
// to see why any cache write was skipped.
const nonSuccessStatusReason = "upstream_non_2xx"

// shouldSkipCacheWriteForStatus reports whether the cache write must be skipped
// because the upstream response was not a 2xx success. Caching a non-2xx body
// (4xx/5xx error envelope, 3xx redirect, 1xx) poisons the cache: a later
// semantically-similar request hits the entry and the stored error body is
// replayed to the client as an HTTP 200 success (see CreateCacheHitResponse in
// req_filter_cache.go). A transient upstream error (429/502/503/504) would
// otherwise be frozen for the whole TTL and served as success even after the
// upstream recovers, defeating client retry logic that keys on the status code.
//
// A zero UpstreamStatusCode means the status was never observed for this
// request (e.g. response headers were not processed, or a unit test drives the
// write path directly). In that case the write is NOT blocked, preserving the
// prior default and avoiding false negatives.
func shouldSkipCacheWriteForStatus(ctx *RequestContext) (bool, string) {
	if ctx == nil || ctx.UpstreamStatusCode == 0 {
		return false, ""
	}
	if ctx.UpstreamStatusCode >= 200 && ctx.UpstreamStatusCode < 300 {
		return false, ""
	}
	return true, nonSuccessStatusReason
}
