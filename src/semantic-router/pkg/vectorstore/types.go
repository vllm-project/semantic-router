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

// Package vectorstore provides types and interfaces for OpenAI-compatible
// Vector Stores API, enabling document ingestion, chunking, embedding,
// and similarity search backed by local storage and embeddings.
package vectorstore

import (
	"fmt"

	"github.com/google/uuid"
)

// VectorStore represents a named collection of embedded documents.
type VectorStore struct {
	ID           string                 `json:"id"`
	Object       string                 `json:"object"` // "vector_store"
	Name         string                 `json:"name"`
	CreatedAt    int64                  `json:"created_at"`
	Status       string                 `json:"status"` // "active", "expired"
	FileCounts   FileCounts             `json:"file_counts"`
	ExpiresAfter *ExpirationPolicy      `json:"expires_after,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	BackendType  string                 `json:"backend_type"` // "milvus", "memory"
}

// FileCounts tracks the processing status of files in a vector store.
type FileCounts struct {
	InProgress int `json:"in_progress"`
	Completed  int `json:"completed"`
	Failed     int `json:"failed"`
	Total      int `json:"total"`
}

// VectorStoreFile represents a file attached to a vector store.
type VectorStoreFile struct {
	ID               string            `json:"id"`
	Object           string            `json:"object"` // "vector_store.file"
	VectorStoreID    string            `json:"vector_store_id"`
	FileID           string            `json:"file_id"`
	Status           string            `json:"status"` // "in_progress", "completed", "failed"
	ChunkingStrategy *ChunkingStrategy `json:"chunking_strategy,omitempty"`
	CreatedAt        int64             `json:"created_at"`
	LastError        *FileError        `json:"last_error,omitempty"`
}

// FileRecord represents an uploaded file.
type FileRecord struct {
	ID        string `json:"id"`
	Object    string `json:"object"` // "file"
	Bytes     int64  `json:"bytes"`
	CreatedAt int64  `json:"created_at"`
	Filename  string `json:"filename"`
	Purpose   string `json:"purpose"`
	Status    string `json:"status"`
}

// ChunkingStrategy defines how documents are split.
type ChunkingStrategy struct {
	Type   string             `json:"type"` // "auto", "static"
	Static *StaticChunkConfig `json:"static,omitempty"`
}

// StaticChunkConfig holds parameters for fixed-size chunking.
type StaticChunkConfig struct {
	MaxChunkSizeTokens int `json:"max_chunk_size_tokens"` // default 800
	ChunkOverlapTokens int `json:"chunk_overlap_tokens"`  // default 400
}

// SearchResult represents a single search hit.
type SearchResult struct {
	FileID     string  `json:"file_id"`
	Filename   string  `json:"filename"`
	Content    string  `json:"content"`
	Score      float64 `json:"score"`
	ChunkIndex int     `json:"chunk_index"`
}

// EmbeddedChunk is a chunk with its embedding, ready for storage.
type EmbeddedChunk struct {
	ID            string    `json:"id"`
	FileID        string    `json:"file_id"`
	Filename      string    `json:"filename"`
	Content       string    `json:"content"`
	Embedding     []float32 `json:"embedding"`
	ChunkIndex    int       `json:"chunk_index"`
	VectorStoreID string    `json:"vector_store_id"`
}

// ExpirationPolicy defines when a vector store should expire.
type ExpirationPolicy struct {
	Anchor string `json:"anchor"` // "last_active_at"
	Days   int    `json:"days"`
}

// FileError describes an error that occurred during file processing.
type FileError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// TextChunk represents a chunk of text extracted from a document.
type TextChunk struct {
	Content    string `json:"content"`
	ChunkIndex int    `json:"chunk_index"`
}

// ID prefix constants for generated identifiers.
const (
	PrefixVectorStore     = "vs_"
	PrefixFile            = "file_"
	PrefixVectorStoreFile = "vsf_"
)

// GenerateVectorStoreID generates a unique ID with the "vs_" prefix.
func GenerateVectorStoreID() string {
	return fmt.Sprintf("%s%s", PrefixVectorStore, uuid.New().String())
}

// GenerateFileID generates a unique ID with the "file_" prefix.
func GenerateFileID() string {
	return fmt.Sprintf("%s%s", PrefixFile, uuid.New().String())
}

// GenerateVectorStoreFileID generates a unique ID with the "vsf_" prefix.
func GenerateVectorStoreFileID() string {
	return fmt.Sprintf("%s%s", PrefixVectorStoreFile, uuid.New().String())
}
