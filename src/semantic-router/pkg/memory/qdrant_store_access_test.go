package memory

import (
	"context"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
)

// fakePointsServer is an in-process Qdrant Points gRPC service: a small payload
// store with conditional (filter-evaluated) SetPayload semantics, enough to
// exercise the store's reinforcement path — including cross-client races —
// without an external service.
type fakePointsServer struct {
	qdrant.UnimplementedPointsServer

	mu       sync.Mutex
	payloads map[string]map[string]*qdrant.Value // keyed by point UUID
	scored   []*qdrant.ScoredPoint               // returned by Query

	setPayload []setPayloadCall

	// gateArmed, when true with a gate channel set, parks the next SetPayload
	// call until the channel is closed; gateEntered is signalled when a call
	// starts waiting.
	gate        chan struct{}
	gateArmed   atomic.Bool
	gateEntered chan struct{}
}

type setPayloadCall struct {
	appliedIDs []string // points the selector actually matched and mutated
	payload    map[string]*qdrant.Value
	wait       bool
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
		// Copy: the response marshals outside this lock while other SetPayload
		// calls mutate the stored map.
		result = append(result, &qdrant.RetrievedPoint{Id: id, Payload: copyPayload(payload)})
	}
	return &qdrant.GetResponse{Result: result}, nil
}

func copyPayload(payload map[string]*qdrant.Value) map[string]*qdrant.Value {
	out := make(map[string]*qdrant.Value, len(payload))
	for key, value := range payload {
		out[key] = value
	}
	return out
}

func (f *fakePointsServer) SetPayload(_ context.Context, req *qdrant.SetPayloadPoints) (*qdrant.PointsOperationResponse, error) {
	// The gate, when armed, holds exactly one SetPayload call — the
	// cross-client race test parks that call there while another client
	// reinforces the same memory.
	if f.gateArmed.CompareAndSwap(true, false) {
		select {
		case f.gateEntered <- struct{}{}:
		default:
		}
		<-f.gate
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	call := setPayloadCall{payload: req.GetPayload(), wait: req.GetWait()}
	for uuid, target := range f.payloads {
		if f.selectorMatches(uuid, target, req.GetPointsSelector()) {
			call.appliedIDs = append(call.appliedIDs, uuid)
			for key, value := range req.GetPayload() {
				target[key] = value
			}
		}
	}
	f.setPayload = append(f.setPayload, call)
	return &qdrant.PointsOperationResponse{Result: &qdrant.UpdateResult{Status: qdrant.UpdateStatus_Completed}}, nil
}

// selectorMatches evaluates the selector the way real Qdrant does for the
// conditions the reinforcement uses: explicit point IDs always match, and a
// filter matches on HasID plus exact integer payload values.
func (f *fakePointsServer) selectorMatches(uuid string, payload map[string]*qdrant.Value, selector *qdrant.PointsSelector) bool {
	if ids := selector.GetPoints().GetIds(); ids != nil {
		for _, id := range ids {
			if id.GetUuid() == uuid {
				return true
			}
		}
		return false
	}
	filter := selector.GetFilter()
	if filter == nil {
		return false
	}
	for _, condition := range filter.GetMust() {
		if hasID := condition.GetHasId(); hasID != nil {
			found := false
			for _, id := range hasID.GetHasId() {
				if id.GetUuid() == uuid {
					found = true
					break
				}
			}
			if !found {
				return false
			}
			continue
		}
		if field := condition.GetField(); field != nil {
			match, ok := payload[field.GetKey()]
			if !ok || match.GetIntegerValue() != field.GetMatch().GetInteger() {
				return false
			}
			continue
		}
		return false
	}
	return true
}

func (f *fakePointsServer) DeletePayload(_ context.Context, req *qdrant.DeletePayloadPoints) (*qdrant.PointsOperationResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for uuid, target := range f.payloads {
		if f.selectorMatches(uuid, target, req.GetPointsSelector()) {
			for _, key := range req.GetKeys() {
				delete(target, key)
			}
		}
	}
	return &qdrant.PointsOperationResponse{Result: &qdrant.UpdateResult{Status: qdrant.UpdateStatus_Completed}}, nil
}

func (f *fakePointsServer) Query(_ context.Context, _ *qdrant.QueryPoints) (*qdrant.QueryResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	result := make([]*qdrant.ScoredPoint, 0, len(f.scored))
	for _, scored := range f.scored {
		result = append(result, &qdrant.ScoredPoint{
			Id:      scored.GetId(),
			Payload: copyPayload(scored.GetPayload()),
			Score:   scored.GetScore(),
			Version: scored.GetVersion(),
		})
	}
	return &qdrant.QueryResponse{Result: result}, nil
}

func (f *fakePointsServer) setPayloadCalls() []setPayloadCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]setPayloadCall(nil), f.setPayload...)
}

func (f *fakePointsServer) accessCountOf(uuid string) int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	if payload, ok := f.payloads[uuid]; ok {
		return payload["access_count"].GetIntegerValue()
	}
	return -1
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
	require.Equal(t, []string{arbitraryIDToUUID("mem-1").GetUuid()}, calls[0].appliedIDs,
		"the compare-and-set must match the point at its observed count")
	require.True(t, calls[0].wait, "reinforcement should wait for the write to apply")
	require.Equal(t, int64(3), calls[0].payload["access_count"].GetIntegerValue())
	var claimFields int
	for key := range calls[0].payload {
		if strings.HasPrefix(key, "access_claim_") {
			claimFields++
		}
	}
	require.Equal(t, 1, claimFields, "the write must carry the per-attempt claim field the caller verifies")
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
		require.Len(t, call.appliedIDs, 1)
		byID[call.appliedIDs[0]] = call.payload
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

// TestQdrantReinforcementSurvivesCrossClientRace reproduces the shared-store
// race deterministically: client A observes the count, then another client's
// reinforcement lands before A's write applies. The unconditional write the
// review flagged would overwrite B's increment; the compare-and-set must lose
// the race, retry, and persist A's own increment too.
func TestQdrantReinforcementSurvivesCrossClientRace(t *testing.T) {
	shared := arbitraryIDToUUID("shared").GetUuid()
	gate := make(chan struct{})
	gateEntered := make(chan struct{}, 4)
	fake := &fakePointsServer{
		payloads: map[string]map[string]*qdrant.Value{
			shared: seededPayload(t, "shared", 5),
		},
		gate:        gate,
		gateArmed:   atomic.Bool{},
		gateEntered: gateEntered,
	}
	fake.gateArmed.Store(true)
	storeA := newReinforcementTestStore(t, fake)
	storeB := newReinforcementTestStore(t, fake)

	aErr := make(chan error, 1)
	go func() { aErr <- storeA.recordRetrieval(context.Background(), "shared") }()

	select {
	case <-gateEntered:
	case <-time.After(5 * time.Second):
		t.Fatal("client A never reached its reinforcement write")
	}

	// Client B reinforces the same memory while A's write is held at the gate.
	require.NoError(t, storeB.recordRetrieval(context.Background(), "shared"))
	require.Equal(t, int64(6), fake.accessCountOf(shared))

	// Release A: its stale compare-and-set must no-op, and its retry must
	// persist the second increment.
	close(gate)
	require.NoError(t, <-aErr)
	require.Equal(t, int64(7), fake.accessCountOf(shared),
		"both clients' reinforcements must persist")
}

// TestQdrantConcurrentReinforcementKeepsEveryIncrement hammers one memory from
// two stores sharing the fake server; every increment must land.
func TestQdrantConcurrentReinforcementKeepsEveryIncrement(t *testing.T) {
	shared := arbitraryIDToUUID("shared").GetUuid()
	fake := &fakePointsServer{
		payloads: map[string]map[string]*qdrant.Value{
			shared: seededPayload(t, "shared", 0),
		},
	}
	storeA := newReinforcementTestStore(t, fake)
	storeB := newReinforcementTestStore(t, fake)

	const perClient = 25
	var wg sync.WaitGroup
	wg.Add(2)
	for _, store := range []*QdrantStore{storeA, storeB} {
		go func(store *QdrantStore) {
			defer wg.Done()
			for range perClient {
				if err := store.recordRetrieval(context.Background(), "shared"); err != nil {
					t.Errorf("reinforcement failed: %v", err)
				}
			}
		}(store)
	}
	wg.Wait()

	require.Equal(t, int64(2*perClient), fake.accessCountOf(shared))
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
