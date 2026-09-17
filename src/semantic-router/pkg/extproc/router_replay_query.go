package extproc

import (
	"errors"
	"math"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func (r *OpenAIRouter) routerReplayReaders() []*routerreplay.Recorder {
	if r.ReplayStoreShared && r.ReplayRecorder != nil {
		return []*routerreplay.Recorder{r.ReplayRecorder}
	}
	seen := make(map[*routerreplay.Recorder]bool)
	readers := make([]*routerreplay.Recorder, 0, len(r.ReplayRecorders))
	keys := make([]string, 0, len(r.ReplayRecorders))
	for key := range r.ReplayRecorders {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		reader := r.ReplayRecorders[key]
		if reader != nil && !seen[reader] {
			readers = append(readers, reader)
			seen[reader] = true
		}
	}
	if len(readers) == 0 && r.ReplayRecorder != nil {
		readers = append(readers, r.ReplayRecorder)
	}
	return readers
}

func (filters routerReplayFilters) storageFilters() store.QueryFilters {
	return store.QueryFilters{
		Search: filters.search, Recipe: filters.recipe, Decision: filters.decision,
		Model: filters.model, CacheStatus: filters.cacheStatus, SessionID: filters.sessionID,
	}
}

func (r *OpenAIRouter) queryRouterReplayPage(query routerReplayListQuery) (routerReplayListResponse, error) {
	readers := r.routerReplayReaders()
	if len(readers) == 1 {
		page, supported, err := readers[0].QueryPage(query.filters.storageFilters(), query.limit, query.offset, query.showDetails)
		if err != nil {
			return routerReplayListResponse{}, err
		}
		if supported {
			return routerReplayPagePayload(page, query), nil
		}
	}
	if query.offset > math.MaxInt-query.limit {
		return routerReplayListResponse{}, errors.New("replay offset is too large")
	}
	// Isolated stores each contribute at most the requested prefix. Database
	// prefixes contain summaries only; bodies are fetched for the final page.
	type sourcedRecord struct {
		record routerreplay.RoutingRecord
		reader *routerreplay.Recorder
	}
	var sourced []sourcedRecord
	total := 0
	for _, reader := range readers {
		page, supported, err := reader.QueryPage(query.filters.storageFilters(), query.offset+query.limit, 0, false)
		if err != nil {
			return routerReplayListResponse{}, err
		}
		if !supported {
			all, listErr := reader.ListRecords()
			if listErr != nil {
				return routerReplayListResponse{}, listErr
			}
			page.Records = filterRouterReplayRecords(all, query.filters)
			page.Total = len(page.Records)
		}
		for _, record := range page.Records {
			sourced = append(sourced, sourcedRecord{record: record, reader: reader})
		}
		total += page.Total
	}
	sort.SliceStable(sourced, func(i, j int) bool {
		if sourced[i].record.Timestamp.Equal(sourced[j].record.Timestamp) {
			return sourced[i].record.ID > sourced[j].record.ID
		}
		return sourced[i].record.Timestamp.After(sourced[j].record.Timestamp)
	})
	records := make([]routerreplay.RoutingRecord, len(sourced))
	for index, source := range sourced {
		records[index] = source.record
	}
	payload := buildRouterReplayListPayload(records, query)
	payload.Total = total
	payload.HasMore = payload.Offset+payload.Count < total
	payload.NextOffset = nil
	if payload.HasMore {
		next := payload.Offset + payload.Count
		payload.NextOffset = &next
	}
	if query.showDetails {
		for index, record := range payload.Data {
			full, found, readErr := sourced[payload.Offset+index].reader.GetRecordWithError(record.ID)
			if readErr != nil {
				return routerReplayListResponse{}, readErr
			}
			if !found {
				return routerReplayListResponse{}, errors.New("replay page record is no longer available")
			}
			payload.Data[index] = full
		}
	}
	return payload, nil
}

func routerReplayPagePayload(page store.RecordPage, query routerReplayListQuery) routerReplayListResponse {
	payload := routerReplayListResponse{
		Object: "router_replay.list", Count: len(page.Records), Total: page.Total,
		Limit: query.limit, Offset: page.Offset, HasMore: page.Offset+len(page.Records) < page.Total,
		Data: page.Records,
	}
	if !query.showDetails {
		for index, record := range payload.Data {
			payload.Data[index] = routerreplay.ListSummaryRecord(record)
		}
	}
	if payload.HasMore {
		next := payload.Offset + payload.Count
		payload.NextOffset = &next
	}
	return payload
}

// Session detail queries use one database snapshot, rather than independent
// offset pages that could duplicate or miss turns during concurrent writes.
func (r *OpenAIRouter) queryRouterReplaySession(sessionID string, recipe *string) ([]routerreplay.RoutingRecord, error) {
	var records []routerreplay.RoutingRecord
	filters := store.QueryFilters{SessionID: sessionID}
	if recipe != nil {
		filters.Recipe = *recipe
		filters.RecipeSet = true
	}
	for _, reader := range r.routerReplayReaders() {
		page, supported, err := reader.QueryPage(filters, math.MaxInt, 0, true)
		if err != nil {
			return nil, err
		}
		if !supported {
			all, listErr := reader.ListRecords()
			if listErr != nil {
				return nil, listErr
			}
			page.Records = filterTrajectoryRecordsBySession(all, sessionID)
			if recipe != nil {
				page.Records = filterTrajectoryRecordsByRecipe(page.Records, *recipe)
			}
		}
		records = append(records, page.Records...)
	}
	return sortRouterReplayRecords(records), nil
}
