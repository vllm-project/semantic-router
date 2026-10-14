package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

type doc struct {
	id, fileID, filename, content string
}

var sampleDocs = []doc{
	// Europe
	{"c1", "f1", "france.txt", "The capital of France is Paris. It is known for the Eiffel Tower."},
	{"c2", "f1", "france.txt", "France is a country in Western Europe with a rich cultural heritage."},
	{"c3", "f2", "germany.txt", "Berlin is the capital of Germany. It is famous for the Brandenburg Gate."},
	{"c4", "f2", "germany.txt", "Germany is the largest economy in Europe and a leader in engineering."},
	// Asia
	{"c5", "f3", "japan.txt", "Tokyo is the capital of Japan. It is one of the most populous cities in the world."},
	{"c6", "f3", "japan.txt", "Japan is an island nation in East Asia known for its technology and cuisine."},
	{"c7", "f4", "india.txt", "New Delhi is the capital of India. Mumbai is the most populated city in India."},
	{"c8", "f4", "india.txt", "India is the most populous country in the world with over 1.4 billion people."},
	{"c9", "f5", "china.txt", "Beijing is the capital of China. Shanghai is the largest city by population."},
	{"c10", "f5", "china.txt", "China has the second largest economy in the world and a long history of innovation."},
}

// runtimeEmbedding serves the Vela embedding model from a model runtime this
// example starts (VLLM_SRUN_COMMAND, else vllm-srun on PATH).
func runtimeEmbedding(ctx context.Context) (*serving.EmbeddingProvider, func(), error) {
	deployment, err := config.ImplicitModelRuntimeDeployment("models/Vela-1.0-Encoder-307M-Embedding", true)
	if err != nil {
		return nil, nil, err
	}
	deployment = deployment.WithDefaults()
	manager := modelservice.NewManager()
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{"embedding": deployment})
	if err != nil {
		_ = manager.Shutdown(ctx)
		return nil, nil, err
	}
	stop := func() {
		_ = lease.Close()
		_ = manager.Shutdown(context.Background())
	}
	provider, err := serving.New(lease, nil).Embedding(ctx, config.ResolvedModelBinding{
		Recipe:     "example",
		Name:       "embedding",
		Binding:    config.ModelBinding{Deployment: "embedding", Contract: "embedding.v1"},
		Deployment: deployment,
	}, 0, 0)
	if err != nil {
		stop()
		return nil, nil, err
	}
	return provider, func() { _ = provider.Close(); stop() }, nil
}

func main() {
	fmt.Println("Valkey Vector Store Backend Example")
	fmt.Println("====================================")

	ctx := context.Background()
	fmt.Println("\n1. Starting the model runtime for embeddings...")
	embedder, stopRuntime, err := runtimeEmbedding(ctx)
	if err != nil {
		log.Fatalf("Failed to start the embedding model: %v", err)
	}
	defer stopRuntime()
	fmt.Println("✓ Embedding model ready")

	backend := initBackend()
	defer backend.Close()

	storeID := fmt.Sprintf("demo_%d", time.Now().UnixNano())
	createCollection(ctx, backend, storeID, embedder.Dimension())
	defer cleanupCollection(ctx, backend, storeID)

	embedAndInsert(ctx, backend, embedder, storeID)
	time.Sleep(500 * time.Millisecond)
	runSearches(ctx, backend, embedder, storeID)
	runFilteredSearch(ctx, backend, embedder, storeID)

	fmt.Println("\n✓ Example completed successfully!")
}

func initBackend() *vectorstore.ValkeyBackend {
	fmt.Println("\n2. Connecting to Valkey...")
	backend, err := vectorstore.NewValkeyBackend(vectorstore.ValkeyBackendConfig{
		Host:             "localhost",
		Port:             6379,
		CollectionPrefix: "example_vs_",
		MetricType:       "COSINE",
		ConnectTimeout:   5,
	})
	if err != nil {
		log.Fatalf("Failed to connect to Valkey: %v", err)
	}
	fmt.Println("✓ Connected to Valkey")
	return backend
}

func createCollection(ctx context.Context, backend *vectorstore.ValkeyBackend, storeID string, dimension int) {
	fmt.Printf("\n3. Creating collection %q (dimension=%d)...\n", storeID, dimension)
	if err := backend.CreateCollection(ctx, storeID, dimension); err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}
	fmt.Println("✓ Collection created")
}

func cleanupCollection(ctx context.Context, backend *vectorstore.ValkeyBackend, storeID string) {
	fmt.Printf("\n7. Cleaning up collection %q...\n", storeID)
	if err := backend.DeleteCollection(ctx, storeID); err != nil {
		log.Printf("Warning: cleanup failed: %v", err)
	} else {
		fmt.Println("✓ Collection deleted")
	}
}

func embedAndInsert(ctx context.Context, backend *vectorstore.ValkeyBackend, embedder *serving.EmbeddingProvider, storeID string) {
	fmt.Println("\n4. Embedding and inserting documents...")
	contents := make([]string, len(sampleDocs))
	for i, d := range sampleDocs {
		contents[i] = d.content
	}
	embeddings, err := embedder.EmbedBatch(ctx, contents)
	if err != nil {
		log.Fatalf("Failed to embed documents: %v", err)
	}
	chunks := make([]vectorstore.EmbeddedChunk, 0, len(sampleDocs))
	for i, d := range sampleDocs {
		chunks = append(chunks, vectorstore.EmbeddedChunk{
			ID: d.id, FileID: d.fileID, Filename: d.filename,
			Content: d.content, Embedding: embeddings[i],
			ChunkIndex: i, VectorStoreID: storeID,
		})
	}
	if err := backend.InsertChunks(ctx, storeID, chunks); err != nil {
		log.Fatalf("Failed to insert chunks: %v", err)
	}
	fmt.Printf("✓ Inserted %d chunks\n", len(chunks))
}

func runSearches(ctx context.Context, backend *vectorstore.ValkeyBackend, embedder *serving.EmbeddingProvider, storeID string) {
	fmt.Println("\n5. Searching for similar documents (threshold=0.80)...")
	for _, query := range []string{
		"What is the capital of France?",
		"Tell me about German engineering",
		"Most populated city in Asia",
	} {
		fmt.Printf("\n  Query: %q\n", query)
		qEmb, err := embedder.Embed(ctx, query)
		if err != nil {
			log.Fatalf("Failed to embed query: %v", err)
		}
		results, err := backend.Search(ctx, storeID, qEmb, 3, 0.80, nil)
		if err != nil {
			log.Fatalf("Search failed: %v", err)
		}
		if len(results) == 0 {
			fmt.Println("    (no results above threshold)")
		}
		for rank, r := range results {
			fmt.Printf("    #%d [%.4f] %s: %s\n", rank+1, r.Score, r.Filename, truncate(r.Content, 70))
		}
	}
}

func runFilteredSearch(ctx context.Context, backend *vectorstore.ValkeyBackend, embedder *serving.EmbeddingProvider, storeID string) {
	fmt.Println("\n6. Searching with file_id filter (only germany.txt)...")
	qEmb, err := embedder.Embed(ctx, "capital city")
	if err != nil {
		log.Fatalf("Failed to embed query: %v", err)
	}
	results, err := backend.Search(ctx, storeID, qEmb, 5, 0.0, map[string]interface{}{"file_id": "f2"})
	if err != nil {
		log.Fatalf("Filtered search failed: %v", err)
	}
	for rank, r := range results {
		fmt.Printf("    #%d [%.4f] %s: %s\n", rank+1, r.Score, r.Filename, truncate(r.Content, 70))
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
