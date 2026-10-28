package routerreplay

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"

// QueryPage uses backend pagination when available. The boolean distinguishes
// unsupported queries from failures; callers must not turn a failure into an
// empty successful response.
func (r *Recorder) QueryPage(filters store.QueryFilters, limit, offset int, details bool) (store.RecordPage, bool, error) {
	reader, ok := r.storage.(store.QueryReader)
	if !ok {
		return store.RecordPage{}, false, nil
	}
	ctx, cancel := r.replayOperationContext()
	defer cancel()
	page, err := reader.QueryPage(ctx, filters, limit, offset, details)
	return page, true, err
}

// ScanMetadata visits every retained record, without retaining captured bodies
// on database backends. Bounded stores keep their existing List implementation.
func (r *Recorder) ScanMetadata(visit func(RoutingRecord) error) error {
	ctx, cancel := r.replayOperationContext()
	defer cancel()
	if reader, ok := r.storage.(store.QueryReader); ok {
		return reader.ScanMetadata(ctx, visit)
	}
	records, err := r.storage.List(ctx)
	if err != nil {
		return err
	}
	for _, record := range records {
		if err := visit(record); err != nil {
			return err
		}
	}
	return nil
}

// GetRecordWithError preserves storage failures for management API readers.
func (r *Recorder) GetRecordWithError(id string) (RoutingRecord, bool, error) {
	return r.getRecord(id)
}
