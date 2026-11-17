package routerreplay

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// QueryPage uses backend pagination when available. The boolean distinguishes
// unsupported queries from failures; callers must not turn a failure into an
// empty successful response.
func (r *Recorder) QueryPage(ctx context.Context, filters store.QueryFilters, limit, offset int, details bool) (store.RecordPage, bool, error) {
	reader, ok := r.storage.(store.QueryReader)
	if !ok {
		return store.RecordPage{}, false, nil
	}
	ctx, cancel := r.replayReadContext(ctx)
	defer cancel()
	page, err := reader.QueryPage(ctx, filters, limit, offset, details)
	return page, true, err
}

// ScanMetadata visits the retained records without their captured bodies.
// Database backends visit only the records the filters select; bounded stores
// visit every record, so callers still apply the filters themselves.
func (r *Recorder) ScanMetadata(ctx context.Context, filters store.QueryFilters, visit func(RoutingRecord) error) error {
	ctx, cancel := r.replayReadContext(ctx)
	defer cancel()
	if reader, ok := r.storage.(store.QueryReader); ok {
		return reader.ScanMetadata(ctx, filters, visit)
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

// ScanFilterOptions visits the values the replay filters offer, from every
// retained record, when ScanMetadata with these filters would not visit every
// record. It reports false, visiting nothing, when that scan already does.
func (r *Recorder) ScanFilterOptions(ctx context.Context, filters store.QueryFilters, visit func(RoutingRecord) error) (bool, error) {
	reader, ok := r.storage.(store.QueryReader)
	if !ok || filters == (store.QueryFilters{}) {
		return false, nil
	}
	ctx, cancel := r.replayReadContext(ctx)
	defer cancel()
	return true, reader.ScanFilterOptions(ctx, visit)
}

// GetRecordWithError preserves storage failures for management API readers.
func (r *Recorder) GetRecordWithError(id string) (RoutingRecord, bool, error) {
	return r.getRecord(id)
}
