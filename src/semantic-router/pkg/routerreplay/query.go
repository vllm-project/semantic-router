package routerreplay

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"

// QueryPage reads one page through the storage backend.
func (r *Recorder) QueryPage(filters store.QueryFilters, limit, offset int, details bool) (store.RecordPage, error) {
	ctx, cancel := r.replayOperationContext()
	defer cancel()
	return r.storage.QueryPage(ctx, filters, limit, offset, details)
}

// ScanMetadata visits every retained record without loading the whole store.
func (r *Recorder) ScanMetadata(visit func(RoutingRecord) error) error {
	ctx, cancel := r.replayOperationContext()
	defer cancel()
	return r.storage.ScanMetadata(ctx, visit)
}

// GetRecordWithError preserves storage failures for management API readers.
func (r *Recorder) GetRecordWithError(id string) (RoutingRecord, bool, error) {
	return r.getRecord(id)
}
