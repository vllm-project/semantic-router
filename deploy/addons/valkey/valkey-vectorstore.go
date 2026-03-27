package main

import (
	"context"
	"fmt"
	"log"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func main() {
	fmt.Println("Valkey Vector Store Backend Example")
	fmt.Println("====================================")

	ctx := context.Background()

	// 1. Initialize the embedding model
	fmt.Println("\n1. Initializing embedding model...")
	err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true)
	if err != nil {
		log.Fatalf("Failed to initialize embedding model: %v", err)
	}
	fmt.Println("✓ Embedding model initialized")

	// 2. Connect to Valkey
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
	defer backend.Close()
	fmt.Println("✓ Connected to Valkey")

	// 3. Create a collection
	storeID := fmt.Sprintf("demo_%d", time.Now().UnixNano())
	dimension := 384 // all-MiniLM-L6-v2 dimension
	fmt.Printf("\n3. Creating collection %q (dimension=%d)...\n", storeID, dimension)
	err = backend.CreateCollection(ctx, storeID, dimension)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}
	fmt.Println("✓ Collection created")

	// Clean up on exit
	defer func() {
		fmt.Printf("\n7. Cleaning up collection %q...\n", storeID)
		if delErr := backend.DeleteCollection(ctx, storeID); delErr != nil {
			log.Printf("Warning: cleanup failed: %v", delErr)
		} else {
			fmt.Println("✓ Collection deleted")
		}
	}()

	// 4. Embed and insert documents
	fmt.Println("\n4. Embedding and inserting documents...")
	docs := []struct {
		id       string
		fileID   string
		filename string
		content  string
	}{
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

	chunks := make([]vectorstore.EmbeddedChunk, 0, len(docs))
	for i, doc := range docs {
		embedding, embErr := candle_binding.GetEmbedding(doc.content, 0)
		if embErr != nil {
			log.Fatalf("Failed to embed document %d: %v", i, embErr)
		}
		chunks = append(chunks, vectorstore.EmbeddedChunk{
			ID:            doc.id,
			FileID:        doc.fileID,
			Filename:      doc.filename,
			Content:       doc.content,
			Embedding:     embedding,
			ChunkIndex:    i,
			VectorStoreID: storeID,
		})
	}

	err = backend.InsertChunks(ctx, storeID, chunks)
	if err != nil {
		log.Fatalf("Failed to insert chunks: %v", err)
	}
	fmt.Printf("✓ Inserted %d chunks\n", len(chunks))

	// Allow indexing time
	time.Sleep(500 * time.Millisecond)

	// 5. Search for similar documents (with threshold to filter low-relevance noise)
	fmt.Println("\n5. Searching for similar documents (threshold=0.80)...")
	queries := []string{
		"What is the capital of France?",
		"Tell me about German engineering",
		"Most populated city in Asia",
	}

	for _, query := range queries {
		fmt.Printf("\n  Query: %q\n", query)
		qEmb, embErr := candle_binding.GetEmbedding(query, 0)
		if embErr != nil {
			log.Fatalf("Failed to embed query: %v", embErr)
		}

		results, searchErr := backend.Search(ctx, storeID, qEmb, 3, 0.80, nil)
		if searchErr != nil {
			log.Fatalf("Search failed: %v", searchErr)
		}

		if len(results) == 0 {
			fmt.Println("    (no results above threshold)")
		}
		for rank, r := range results {
			fmt.Printf("    #%d [%.4f] %s: %s\n", rank+1, r.Score, r.Filename, truncate(r.Content, 70))
		}
	}

	// 6. Search with file_id filter
	fmt.Println("\n6. Searching with file_id filter (only germany.txt)...")
	qEmb, err := candle_binding.GetEmbedding("capital city", 0)
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

	fmt.Println("\n✓ Example completed successfully!")
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
