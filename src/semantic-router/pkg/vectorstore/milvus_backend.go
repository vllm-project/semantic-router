/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vectorstore

import (
	"context"
	"fmt"
	"regexp"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// safeIdentifierPattern matches UUIDs, prefixed IDs (file_xxx, vs_xxx), and simple alphanumeric strings.
// Used to prevent Milvus expression injection in filter values.
var safeIdentifierPattern = regexp.MustCompile(`^[a-zA-Z0-9_\-]+$`)

// MilvusBackendConfig holds configuration for the Milvus backend.
type MilvusBackendConfig struct {
	Address          string // host:port
	CollectionPrefix string // prefix for collection names (default "vsr_vs_")
	IndexM           int    // HNSW M parameter (default 16)
	IndexEf          int    // HNSW efConstruction (default 200)
	SearchEf         int    // search ef parameter (default 64)
	ConnectTimeout   int    // connection timeout in seconds (default 10)
}

// MilvusBackend implements VectorStoreBackend using Milvus.
type MilvusBackend struct {
	client           client.Client
	collectionPrefix string
	indexM           int
	indexEf          int
	searchEf         int
}

// NewMilvusBackend creates a new Milvus vector store backend.
func NewMilvusBackend(cfg MilvusBackendConfig) (*MilvusBackend, error) {
	if cfg.Address == "" {
		return nil, fmt.Errorf("milvus address is required")
	}

	prefix := cfg.CollectionPrefix
	if prefix == "" {
		prefix = "vsr_vs_"
	}
	indexM := cfg.IndexM
	if indexM <= 0 {
		indexM = 16
	}
	indexEf := cfg.IndexEf
	if indexEf <= 0 {
		indexEf = 200
	}
	searchEf := cfg.SearchEf
	if searchEf <= 0 {
		searchEf = 64
	}

	timeout := cfg.ConnectTimeout
	if timeout <= 0 {
		timeout = 10
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	milvusClient, err := client.NewGrpcClient(ctx, cfg.Address)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Milvus at %s: %w", cfg.Address, err)
	}

	return &MilvusBackend{
		client:           milvusClient,
		collectionPrefix: prefix,
		indexM:           indexM,
		indexEf:          indexEf,
		searchEf:         searchEf,
	}, nil
}

func (m *MilvusBackend) collectionName(vectorStoreID string) string {
	return m.collectionPrefix + vectorStoreID
}

// CreateCollection creates a Milvus collection with the schema for document chunks.
func (m *MilvusBackend) CreateCollection(ctx context.Context, vectorStoreID string, dimension int) error {
	colName := m.collectionName(vectorStoreID)

	schema := &entity.Schema{
		CollectionName: colName,
		Description:    fmt.Sprintf("Vector store: %s", vectorStoreID),
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				TypeParams: map[string]string{"max_length": "128"},
			},
			{
				Name:       "file_id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "128"},
			},
			{
				Name:       "filename",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "512"},
			},
			{
				Name:       "content",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:     "embedding",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", dimension),
				},
			},
			{
				Name:     "chunk_index",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:     "created_at",
				DataType: entity.FieldTypeInt64,
			},
		},
	}

	if err := m.client.CreateCollection(ctx, schema, 1); err != nil {
		return fmt.Errorf("failed to create collection %s: %w", colName, err)
	}

	// Create HNSW index on the embedding field.
	index, err := entity.NewIndexHNSW(entity.IP, m.indexM, m.indexEf)
	if err != nil {
		return fmt.Errorf("failed to create HNSW index: %w", err)
	}
	if err := m.client.CreateIndex(ctx, colName, "embedding", index, false); err != nil {
		return fmt.Errorf("failed to create index on %s: %w", colName, err)
	}

	// Load collection into memory.
	if err := m.client.LoadCollection(ctx, colName, false); err != nil {
		return fmt.Errorf("failed to load collection %s: %w", colName, err)
	}

	return nil
}

// DeleteCollection drops a Milvus collection.
func (m *MilvusBackend) DeleteCollection(ctx context.Context, vectorStoreID string) error {
	colName := m.collectionName(vectorStoreID)
	if err := m.client.DropCollection(ctx, colName); err != nil {
		return fmt.Errorf("failed to drop collection %s: %w", colName, err)
	}
	return nil
}

// CollectionExists checks if a Milvus collection exists.
func (m *MilvusBackend) CollectionExists(ctx context.Context, vectorStoreID string) (bool, error) {
	colName := m.collectionName(vectorStoreID)
	exists, err := m.client.HasCollection(ctx, colName)
	if err != nil {
		return false, fmt.Errorf("failed to check collection %s: %w", colName, err)
	}
	return exists, nil
}

// InsertChunks inserts embedded chunks into the Milvus collection.
func (m *MilvusBackend) InsertChunks(ctx context.Context, vectorStoreID string, chunks []EmbeddedChunk) error {
	if len(chunks) == 0 {
		return nil
	}

	colName := m.collectionName(vectorStoreID)
	now := time.Now().Unix()

	ids := make([]string, len(chunks))
	fileIDs := make([]string, len(chunks))
	filenames := make([]string, len(chunks))
	contents := make([]string, len(chunks))
	embeddings := make([][]float32, len(chunks))
	chunkIndices := make([]int64, len(chunks))
	createdAts := make([]int64, len(chunks))

	for i, c := range chunks {
		ids[i] = c.ID
		fileIDs[i] = c.FileID
		filenames[i] = c.Filename
		contents[i] = c.Content
		embeddings[i] = c.Embedding
		chunkIndices[i] = int64(c.ChunkIndex)
		createdAts[i] = now
	}

	dim := len(chunks[0].Embedding)

	_, err := m.client.Upsert(ctx, colName, "",
		entity.NewColumnVarChar("id", ids),
		entity.NewColumnVarChar("file_id", fileIDs),
		entity.NewColumnVarChar("filename", filenames),
		entity.NewColumnVarChar("content", contents),
		entity.NewColumnFloatVector("embedding", dim, embeddings),
		entity.NewColumnInt64("chunk_index", chunkIndices),
		entity.NewColumnInt64("created_at", createdAts),
	)
	if err != nil {
		return fmt.Errorf("failed to insert chunks into %s: %w", colName, err)
	}

	if err := m.client.Flush(ctx, colName, false); err != nil {
		return fmt.Errorf("failed to flush %s: %w", colName, err)
	}

	return nil
}

// DeleteByFileID removes all chunks from a collection that belong to the given file.
func (m *MilvusBackend) DeleteByFileID(ctx context.Context, vectorStoreID string, fileID string) error {
	if !safeIdentifierPattern.MatchString(fileID) {
		return fmt.Errorf("invalid file ID: contains disallowed characters")
	}
	colName := m.collectionName(vectorStoreID)
	expr := fmt.Sprintf("file_id == \"%s\"", fileID)

	if err := m.client.Delete(ctx, colName, "", expr); err != nil {
		return fmt.Errorf("failed to delete chunks for file %s from %s: %w", fileID, colName, err)
	}
	return nil
}

// Search performs vector similarity search in a Milvus collection.
func (m *MilvusBackend) Search(
	ctx context.Context, vectorStoreID string, queryEmbedding []float32,
	topK int, threshold float32, filter map[string]interface{},
) ([]SearchResult, error) {
	colName := m.collectionName(vectorStoreID)

	// Build filter expression with injection prevention.
	expr := ""
	if filter != nil {
		if fid, ok := filter["file_id"].(string); ok && fid != "" {
			if !safeIdentifierPattern.MatchString(fid) {
				return nil, fmt.Errorf("invalid file_id filter: contains disallowed characters")
			}
			expr = fmt.Sprintf("file_id == \"%s\"", fid)
		}
	}

	sp, err := entity.NewIndexHNSWSearchParam(m.searchEf)
	if err != nil {
		return nil, fmt.Errorf("failed to create search params: %w", err)
	}

	searchResults, err := m.client.Search(ctx, colName, []string{}, expr,
		[]string{"file_id", "filename", "content", "chunk_index"},
		[]entity.Vector{entity.FloatVector(queryEmbedding)},
		"embedding", entity.IP, topK, sp)
	if err != nil {
		return nil, fmt.Errorf("failed to search in %s: %w", colName, err)
	}

	var results []SearchResult
	for _, sr := range searchResults {
		for i := 0; i < sr.ResultCount; i++ {
			score := float64(sr.Scores[i])
			if score < float64(threshold) {
				continue
			}

			result := SearchResult{Score: score}

			// Extract field values by column name.
			for _, field := range sr.Fields {
				switch col := field.(type) {
				case *entity.ColumnVarChar:
					val, err := col.ValueByIdx(i)
					if err != nil {
						continue
					}
					switch col.Name() {
					case "file_id":
						result.FileID = val
					case "filename":
						result.Filename = val
					case "content":
						result.Content = val
					}
				case *entity.ColumnInt64:
					val, err := col.ValueByIdx(i)
					if err != nil {
						continue
					}
					if col.Name() == "chunk_index" {
						result.ChunkIndex = int(val)
					}
				}
			}

			results = append(results, result)
		}
	}

	return results, nil
}

// Close releases the Milvus client connection.
func (m *MilvusBackend) Close() error {
	if m.client != nil {
		return m.client.Close()
	}
	return nil
}
