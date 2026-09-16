package store

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// recordingMilvusClient stubs the Milvus calls the replay store issues so
// tests can assert the consistency-level options actually reach the SDK; the
// embedded nil interface covers every other client method.
type recordingMilvusClient struct {
	client.Client
	queryOpts    []client.SearchQueryOptionFunc
	queryResults client.ResultSet
}

func (f *recordingMilvusClient) Query(_ context.Context, _ string, _ []string, _ string, _ []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
	f.queryOpts = opts
	return f.queryResults, nil
}

func (f *recordingMilvusClient) Insert(_ context.Context, _ string, _ string, _ ...entity.Column) (entity.Column, error) {
	return nil, nil
}

func (f *recordingMilvusClient) Flush(_ context.Context, _ string, _ bool, _ ...client.FlushOption) error {
	return nil
}

// appliedConsistencyLevel replays the read options against a sentinel value
// that differs from every mapped level, so "option did nothing" (ClStrong is
// the zero value) cannot masquerade as an explicit Strong.
func appliedConsistencyLevel(t *testing.T, opts []client.SearchQueryOptionFunc) entity.ConsistencyLevel {
	t.Helper()
	applied := &client.SearchQueryOption{ConsistencyLevel: entity.ClCustomized}
	for _, opt := range opts {
		opt(applied)
	}
	return applied.ConsistencyLevel
}

func newRecordingStore(t *testing.T, level string) (*MilvusStore, *recordingMilvusClient) {
	t.Helper()
	consistencyLevel, ok := milvusConsistencyLevel(level)
	if !ok {
		t.Fatalf("milvusConsistencyLevel(%q) reported the level as invalid", level)
	}
	fake := &recordingMilvusClient{}
	return &MilvusStore{
		client:           fake,
		collectionName:   DefaultMilvusCollection,
		consistencyLevel: consistencyLevel,
		lifecycle:        newStorageLifecycle(),
	}, fake
}

func TestMilvusStoreReadsHonorConsistencyLevel(t *testing.T) {
	for _, level := range []string{"Strong", "Bounded", "Eventually"} {
		t.Run(level, func(t *testing.T) {
			store, fake := newRecordingStore(t, level)
			want, _ := milvusConsistencyLevel(level)

			if _, _, err := store.getRecord(context.Background(), "record-1"); err != nil {
				t.Fatalf("getRecord returned %v", err)
			}
			if got := appliedConsistencyLevel(t, fake.queryOpts); got != want {
				t.Fatalf("getRecord applied consistency level %v, want %v", got, want)
			}

			fake.queryOpts = nil
			if _, err := store.listRecords(context.Background()); err != nil {
				t.Fatalf("listRecords returned %v", err)
			}
			if got := appliedConsistencyLevel(t, fake.queryOpts); got != want {
				t.Fatalf("listRecords applied consistency level %v, want %v", got, want)
			}
		})
	}
}

// Session degrades to Eventually in the SDK until the client has a session
// timestamp for the collection, so reads before the first write are floored at
// Bounded to stay at least as fresh as the previous behavior.
func TestMilvusStoreSessionReadsFloorAtBoundedUntilFirstWrite(t *testing.T) {
	store, fake := newRecordingStore(t, "Session")

	if _, _, err := store.getRecord(context.Background(), "record-1"); err != nil {
		t.Fatalf("getRecord returned %v", err)
	}
	if got := appliedConsistencyLevel(t, fake.queryOpts); got != entity.ClBounded {
		t.Fatalf("pre-write read applied consistency level %v, want %v", got, entity.ClBounded)
	}

	if err := store.insertRecord(context.Background(), Record{ID: "record-1"}, []byte("{}")); err != nil {
		t.Fatalf("insertRecord returned %v", err)
	}

	fake.queryOpts = nil
	if _, _, err := store.getRecord(context.Background(), "record-1"); err != nil {
		t.Fatalf("getRecord returned %v", err)
	}
	if got := appliedConsistencyLevel(t, fake.queryOpts); got != entity.ClSession {
		t.Fatalf("post-write read applied consistency level %v, want %v", got, entity.ClSession)
	}
}
