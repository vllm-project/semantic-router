package memory

import (
	"context"
	"errors"
	"net"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type qdrantUpdateTestServer struct {
	qdrant.UnimplementedPointsServer

	mu               sync.Mutex
	existingMemory   *Memory
	upsertRequest    *qdrant.UpsertPoints
	upsertCalls      int
	upsertErr        error
	persistedPayload map[string]*qdrant.Value
}

func (s *qdrantUpdateTestServer) Get(_ context.Context, _ *qdrant.GetPoints) (*qdrant.GetResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return &qdrant.GetResponse{Result: []*qdrant.RetrievedPoint{{
		Id:      arbitraryIDToUUID(s.existingMemory.ID),
		Payload: qdrant.NewValueMap(memoryToPayload(s.existingMemory)),
	}}}, nil
}

func (s *qdrantUpdateTestServer) Upsert(_ context.Context, request *qdrant.UpsertPoints) (*qdrant.PointsOperationResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.upsertCalls++
	s.upsertRequest = request
	if s.upsertErr != nil {
		return nil, s.upsertErr
	}
	s.persistedPayload = request.Points[0].Payload
	return &qdrant.PointsOperationResponse{Result: &qdrant.UpdateResult{Status: qdrant.UpdateStatus_Completed}}, nil
}

func newQdrantUpdateTestStore(t *testing.T, server *qdrantUpdateTestServer, embeddingConfig EmbeddingConfig) *QdrantStore {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	grpcServer := grpc.NewServer()
	qdrant.RegisterPointsServer(grpcServer, server)
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(grpcServer.Stop)

	_, portText, err := net.SplitHostPort(listener.Addr().String())
	require.NoError(t, err)
	port, err := strconv.Atoi(portText)
	require.NoError(t, err)
	client, err := qdrant.NewClient(&qdrant.Config{
		Host:                   "127.0.0.1",
		Port:                   port,
		PoolSize:               1,
		SkipCompatibilityCheck: true,
		KeepAliveTime:          -1,
	})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, client.Close()) })

	return &QdrantStore{
		client:          client,
		collectionName:  "test_memories",
		enabled:         true,
		embeddingConfig: embeddingConfig,
	}
}

func TestQdrantStoreUpdatePreservesCreatedAtInUpsertPayload(t *testing.T) {
	createdAt := time.Unix(1_700_000_000, 0)
	server := &qdrantUpdateTestServer{existingMemory: &Memory{
		ID:        "memory-1",
		Content:   "original content",
		UserID:    "user-1",
		Type:      MemoryTypeSemantic,
		CreatedAt: createdAt,
	}}
	store := newQdrantUpdateTestStore(t, server, EmbeddingConfig{})
	updated := &Memory{
		Content:   "updated content",
		UserID:    "user-1",
		Type:      MemoryTypeSemantic,
		Embedding: []float32{0.25, 0.75},
	}

	require.NoError(t, store.Update(context.Background(), "memory-1", updated))

	server.mu.Lock()
	defer server.mu.Unlock()
	require.Equal(t, 1, server.upsertCalls)
	require.Len(t, server.upsertRequest.Points, 1)
	require.Equal(t, "updated content", server.upsertRequest.Points[0].Payload["content"].GetStringValue())
	require.Equal(t, createdAt.Unix(), server.upsertRequest.Points[0].Payload["created_at"].GetIntegerValue())
	require.Equal(t, createdAt, updated.CreatedAt)
}

func TestQdrantStoreUpdateHonorsExplicitCreatedAt(t *testing.T) {
	server := &qdrantUpdateTestServer{existingMemory: &Memory{
		ID:        "memory-1",
		Content:   "original content",
		UserID:    "user-1",
		CreatedAt: time.Unix(1_700_000_000, 0),
	}}
	store := newQdrantUpdateTestStore(t, server, EmbeddingConfig{})
	explicitCreatedAt := time.Unix(1_710_000_000, 0)
	updated := &Memory{
		Content:   "updated content",
		UserID:    "user-1",
		Embedding: []float32{0.25, 0.75},
		CreatedAt: explicitCreatedAt,
	}

	require.NoError(t, store.Update(context.Background(), "memory-1", updated))

	server.mu.Lock()
	defer server.mu.Unlock()
	require.Equal(t, explicitCreatedAt.Unix(), server.upsertRequest.Points[0].Payload["created_at"].GetIntegerValue())
}

func TestQdrantStoreUpdateEmbeddingFailureDoesNotUpsert(t *testing.T) {
	server := &qdrantUpdateTestServer{existingMemory: &Memory{
		ID:        "memory-1",
		Content:   "original content",
		UserID:    "user-1",
		CreatedAt: time.Unix(1_700_000_000, 0),
	}}
	provider, err := embedding.NewFuncProvider("qdrant-update-test", 2, func(context.Context, string) ([]float32, error) {
		return nil, errors.New("embedding failed")
	})
	require.NoError(t, err)
	store := newQdrantUpdateTestStore(t, server, EmbeddingConfig{Model: EmbeddingModelQwen3, Provider: provider})

	err = store.Update(context.Background(), "memory-1", &Memory{Content: "updated content", UserID: "user-1"})
	require.ErrorContains(t, err, "failed to generate embedding")

	server.mu.Lock()
	defer server.mu.Unlock()
	require.Zero(t, server.upsertCalls)
	require.Nil(t, server.persistedPayload)
}

func TestQdrantStoreUpdateWriteFailureDoesNotPersistPayload(t *testing.T) {
	writeErr := errors.New("qdrant unavailable")
	server := &qdrantUpdateTestServer{
		existingMemory: &Memory{
			ID:        "memory-1",
			Content:   "original content",
			UserID:    "user-1",
			CreatedAt: time.Unix(1_700_000_000, 0),
		},
		upsertErr: writeErr,
	}
	store := newQdrantUpdateTestStore(t, server, EmbeddingConfig{})
	updated := &Memory{
		Content:   "updated content",
		UserID:    "user-1",
		Embedding: []float32{0.25, 0.75},
	}

	err := store.Update(context.Background(), "memory-1", updated)
	require.ErrorContains(t, err, "qdrant update failed")
	require.ErrorContains(t, err, writeErr.Error())

	server.mu.Lock()
	defer server.mu.Unlock()
	require.Equal(t, 1, server.upsertCalls)
	require.Nil(t, server.persistedPayload)
	require.Equal(t, time.Unix(1_700_000_000, 0).Unix(), server.upsertRequest.Points[0].Payload["created_at"].GetIntegerValue())
}
