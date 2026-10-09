package memory

import (
	"context"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
)

// fakePointsServer is an in-process Qdrant Points gRPC service: a small
// payload store plus recorded SetPayload calls, enough to exercise the
// store's reinforcement path without an external service.
type fakePointsServer struct {
	qdrant.UnimplementedPointsServer

	mu         sync.Mutex
	payloads   map[string]map[string]*qdrant.Value // keyed by point UUID
	scored     []*qdrant.ScoredPoint               // returned by Query
	setPayload []setPayloadCall
}

type setPayloadCall struct {
	ids     []string
	payload map[string]*qdrant.Value
	wait    bool
}

func (f *fakePointsServer) Get(_ context.Context, req *qdrant.GetPoints) (*qdrant.GetResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	result := make([]*qdrant.RetrievedPoint, 0, len(req.GetIds()))
	for _, id := range req.GetIds() {
		payload, ok := f.payloads[id.GetUuid()]
		if !ok {
			continue
		}
		result = append(result, &qdrant.RetrievedPoint{Id: id, Payload: payload})
	}
	return &qdrant.GetResponse{Result: result}, nil
}

func (f *fakePointsServer) SetPayload(_ context.Context, req *qdrant.SetPayloadPoints) (*qdrant.PointsOperationResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	call := setPayloadCall{payload: req.GetPayload(), wait: req.GetWait()}
	for _, id := range req.GetPointsSelector().GetPoints().GetIds() {
		call.ids = append(call.ids, id.GetUuid())
		target, ok := f.payloads[id.GetUuid()]
		if !ok {
			continue
		}
		for key, value := range req.GetPayload() {
			target[key] = value
		}
	}
	f.setPayload = append(f.setPayload, call)
	return &qdrant.PointsOperationResponse{Result: &qdrant.UpdateResult{Status: qdrant.UpdateStatus_Completed}}, nil
}

func (f *fakePointsServer) Query(_ context.Context, _ *qdrant.QueryPoints) (*qdrant.QueryResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return &qdrant.QueryResponse{Result: f.scored}, nil
}

func (f *fakePointsServer) setPayloadCalls() []setPayloadCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]setPayloadCall(nil), f.setPayload...)
}

func startFakeQdrant(t *testing.T, fake *fakePointsServer) *qdrant.Client {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	server := grpc.NewServer()
	qdrant.RegisterPointsServer(server, fake)
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(server.Stop)

	client, err := qdrant.NewClient(&qdrant.Config{
		Host:                   "127.0.0.1",
		Port:                   listener.Addr().(*net.TCPAddr).Port,
		SkipCompatibilityCheck: true,
		PoolSize:               1,
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = client.Close() })
	return client
}

// newReinforcementTestStore builds an enabled store against the fake server
// with the deterministic storage fixture as its embedding provider.
func newReinforcementTestStore(t *testing.T, fake *fakePointsServer) *QdrantStore {
	t.Helper()
	return &QdrantStore{
		client:         startFakeQdrant(t, fake),
		collectionName: "reinforcement_test",
		enabled:        true,
		embeddingConfig: EmbeddingConfig{
			Provider:  storageMemoryVectors(),
			Model:     EmbeddingModelQwen3,
			Dimension: 384,
		},
	}
}

// waitForSetPayloadCalls polls until the background reinforcement lands.
func waitForSetPayloadCalls(t *testing.T, fake *fakePointsServer, want int) []setPayloadCall {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		calls := fake.setPayloadCalls()
		if len(calls) >= want {
			return calls
		}
		if time.Now().After(deadline) {
			t.Fatalf("reinforcement never landed: %d SetPayload calls, want %d", len(calls), want)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func seededPayload(t *testing.T, id string, accessCount int) map[string]*qdrant.Value {
	t.Helper()
	base := time.Unix(1_700_000_000, 0)
	return qdrant.NewValueMap(memoryToPayload(&Memory{
		ID:           id,
		Content:      "The user prefers dark mode in all their applications",
		UserID:       "user-a",
		Type:         MemoryTypeSemantic,
		AccessCount:  accessCount,
		CreatedAt:    base,
		UpdatedAt:    base,
		LastAccessed: base,
	}))
}

func TestQdrantRetrieveReinforcesAccessTracking(t *testing.T) {
	fake := &fakePointsServer{
		payloads: map[string]map[string]*qdrant.Value{
			arbitraryIDToUUID("mem-1").GetUuid(): seededPayload(t, "mem-1", 2),
		},
		scored: []*qdrant.ScoredPoint{{
			Id:      arbitraryIDToUUID("mem-1"),
			Score:   1.0,
			Payload: qdrant.NewValueMap(map[string]any{}),
		}},
	}
	// The Query result must carry the point payload so the returned memory has
	// its write-time ID; reuse the stored payload.
	fake.mu.Lock()
	fake.scored[0].Payload = fake.payloads[arbitraryIDToUUID("mem-1").GetUuid()]
	fake.mu.Unlock()

	store := newReinforcementTestStore(t, fake)
	results, err := store.Retrieve(context.Background(), RetrieveOptions{
		Query:  "What are the user's display preferences?",
		UserID: "user-a",
		Limit:  5,
	})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, "mem-1", results[0].Memory.ID)
	require.Equal(t, 2, results[0].Memory.AccessCount, "the read path must not mutate the returned memory")

	calls := waitForSetPayloadCalls(t, fake, 1)
	require.Len(t, calls, 1)
	require.Equal(t, []string{arbitraryIDToUUID("mem-1").GetUuid()}, calls[0].ids)
	require.True(t, calls[0].wait, "reinforcement should wait for the write to apply")
	require.Equal(t, int64(3), calls[0].payload["access_count"].GetIntegerValue())
	require.Greater(t, calls[0].payload["last_accessed"].GetIntegerValue(), int64(1_700_000_000))
	require.Equal(t, calls[0].payload["last_accessed"].GetIntegerValue(), calls[0].payload["updated_at"].GetIntegerValue())
}

func TestQdrantRecordRetrievalBatchIncrementsEachMemory(t *testing.T) {
	fake := &fakePointsServer{
		payloads: map[string]map[string]*qdrant.Value{
			arbitraryIDToUUID("cold").GetUuid(): seededPayload(t, "cold", 0),
			arbitraryIDToUUID("warm").GetUuid(): seededPayload(t, "warm", 5),
		},
	}
	store := newReinforcementTestStore(t, fake)

	store.recordRetrievalBatch([]string{"cold", "warm"})

	byID := map[string]map[string]*qdrant.Value{}
	for _, call := range fake.setPayloadCalls() {
		require.Len(t, call.ids, 1)
		byID[call.ids[0]] = call.payload
	}
	require.Len(t, byID, 2)
	require.Equal(t, int64(1), byID[arbitraryIDToUUID("cold").GetUuid()]["access_count"].GetIntegerValue())
	require.Equal(t, int64(6), byID[arbitraryIDToUUID("warm").GetUuid()]["access_count"].GetIntegerValue())
	for _, payload := range byID {
		require.Greater(t, payload["last_accessed"].GetIntegerValue(), int64(1_700_000_000))
		require.Equal(t, payload["last_accessed"].GetIntegerValue(), payload["updated_at"].GetIntegerValue())
	}
}

func TestQdrantRecordRetrievalMissingMemoryWritesNothing(t *testing.T) {
	fake := &fakePointsServer{payloads: map[string]map[string]*qdrant.Value{}}
	store := newReinforcementTestStore(t, fake)

	store.recordRetrievalBatch([]string{"missing"})

	require.Empty(t, fake.setPayloadCalls())
}

func TestRetrieveResultIDsKeepsResultOrder(t *testing.T) {
	results := []*RetrieveResult{
		{Memory: &Memory{ID: "b"}, Score: 0.9},
		{Memory: &Memory{ID: "a"}, Score: 0.8},
		{Memory: &Memory{ID: "c"}, Score: 0.7},
	}
	require.Equal(t, []string{"b", "a", "c"}, retrieveResultIDs(results))
	require.Empty(t, retrieveResultIDs(nil))
}
