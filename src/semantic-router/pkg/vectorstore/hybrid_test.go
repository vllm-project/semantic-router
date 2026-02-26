package vectorstore

import (
	"context"
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

func TestTokenize(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"Hello World", []string{"hello", "world"}},
		{"BM25-scoring is great!", []string{"bm25", "scoring", "is", "great"}},
		{"", nil},
		{"  spaces  ", []string{"spaces"}},
		{"CamelCase123", []string{"camelcase123"}},
	}
	for _, tt := range tests {
		got := tokenize(tt.input)
		if len(got) != len(tt.want) {
			t.Errorf("tokenize(%q) = %v, want %v", tt.input, got, tt.want)
			continue
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Errorf("tokenize(%q)[%d] = %q, want %q", tt.input, i, got[i], tt.want[i])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// N-gram helpers
// ---------------------------------------------------------------------------

func TestCharNgrams(t *testing.T) {
	ngrams := charNgrams("world", 3)
	expected := map[string]struct{}{
		"wor": {}, "orl": {}, "rld": {},
	}
	if len(ngrams) != len(expected) {
		t.Fatalf("charNgrams(world, 3) produced %d ngrams, want %d", len(ngrams), len(expected))
	}
	for k := range expected {
		if _, ok := ngrams[k]; !ok {
			t.Errorf("missing expected n-gram %q", k)
		}
	}
}

func TestCharNgramsShorterThanN(t *testing.T) {
	ngrams := charNgrams("ab", 3)
	if len(ngrams) != 1 {
		t.Fatalf("expected 1 n-gram for text shorter than n, got %d", len(ngrams))
	}
	if _, ok := ngrams["ab"]; !ok {
		t.Errorf("expected n-gram 'ab' for text shorter than n")
	}
}

func TestJaccardSimilarity(t *testing.T) {
	a := map[string]struct{}{"a": {}, "b": {}, "c": {}}
	b := map[string]struct{}{"b": {}, "c": {}, "d": {}}
	got := jaccardSimilarity(a, b)
	// intersection = {b, c} = 2, union = {a, b, c, d} = 4 -> 0.5
	if math.Abs(got-0.5) > 1e-9 {
		t.Errorf("jaccardSimilarity = %f, want 0.5", got)
	}
}

func TestJaccardIdentical(t *testing.T) {
	a := map[string]struct{}{"x": {}, "y": {}}
	got := jaccardSimilarity(a, a)
	if math.Abs(got-1.0) > 1e-9 {
		t.Errorf("jaccardSimilarity(identical) = %f, want 1.0", got)
	}
}

func TestJaccardDisjoint(t *testing.T) {
	a := map[string]struct{}{"a": {}}
	b := map[string]struct{}{"z": {}}
	got := jaccardSimilarity(a, b)
	if got != 0 {
		t.Errorf("jaccardSimilarity(disjoint) = %f, want 0", got)
	}
}

// ---------------------------------------------------------------------------
// BM25 Index
// ---------------------------------------------------------------------------

func makeTestChunks() map[string]EmbeddedChunk {
	return map[string]EmbeddedChunk{
		"c1": {ID: "c1", Content: "the quick brown fox jumps over the lazy dog", Embedding: []float32{1, 0, 0}},
		"c2": {ID: "c2", Content: "machine learning and deep learning are subfields of artificial intelligence", Embedding: []float32{0, 1, 0}},
		"c3": {ID: "c3", Content: "the fox is quick and brown", Embedding: []float32{0.5, 0.5, 0}},
	}
}

func TestBM25Index_BasicScoring(t *testing.T) {
	chunks := makeTestChunks()
	idx := NewBM25Index(chunks)

	scores := idx.Score("quick brown fox", 1.2, 0.75)
	if len(scores) == 0 {
		t.Fatal("expected BM25 scores for matching query")
	}

	// "quick brown fox" should match c1 and c3, not c2
	if _, ok := scores["c2"]; ok {
		t.Errorf("c2 should not match 'quick brown fox'")
	}
	if scores["c1"] == 0 {
		t.Errorf("c1 should have non-zero BM25 score for 'quick brown fox'")
	}
	if scores["c3"] == 0 {
		t.Errorf("c3 should have non-zero BM25 score for 'quick brown fox'")
	}
}

func TestBM25Index_NoMatch(t *testing.T) {
	chunks := makeTestChunks()
	idx := NewBM25Index(chunks)

	scores := idx.Score("quantum computing", 1.2, 0.75)
	if len(scores) != 0 {
		t.Errorf("expected no BM25 matches for unrelated query, got %d", len(scores))
	}
}

func TestBM25Index_EmptyIndex(t *testing.T) {
	idx := NewBM25Index(map[string]EmbeddedChunk{})
	scores := idx.Score("anything", 1.2, 0.75)
	if scores != nil {
		t.Errorf("expected nil scores for empty index")
	}
}

func TestBM25Index_EmptyQuery(t *testing.T) {
	chunks := makeTestChunks()
	idx := NewBM25Index(chunks)
	scores := idx.Score("", 1.2, 0.75)
	if scores != nil {
		t.Errorf("expected nil scores for empty query")
	}
}

// ---------------------------------------------------------------------------
// N-gram Index
// ---------------------------------------------------------------------------

func TestNgramIndex_BasicScoring(t *testing.T) {
	chunks := makeTestChunks()
	idx := NewNgramIndex(chunks, 3)

	scores := idx.Score("quick brown fox")
	if len(scores) == 0 {
		t.Fatal("expected n-gram scores for overlapping query")
	}

	// c1 ("the quick brown fox...") and c3 ("the fox is quick and brown")
	// should have higher similarity than c2 (machine learning)
	if scores["c1"] <= scores["c2"] {
		t.Errorf("c1 (%f) should score higher than c2 (%f) for 'quick brown fox'",
			scores["c1"], scores["c2"])
	}
}

func TestNgramIndex_NgramN(t *testing.T) {
	chunks := makeTestChunks()
	idx := NewNgramIndex(chunks, 4)
	if idx.NgramN() != 4 {
		t.Errorf("NgramN() = %d, want 4", idx.NgramN())
	}
}

// ---------------------------------------------------------------------------
// Score Fusion
// ---------------------------------------------------------------------------

func TestFuseScores_Weighted(t *testing.T) {
	vectorScores := map[string]float64{"a": 0.9, "b": 0.5, "c": 0.3}
	bm25Scores := map[string]float64{"a": 2.0, "b": 4.0, "d": 1.0}
	ngramScores := map[string]float64{"a": 0.8, "c": 0.6}

	config := &HybridSearchConfig{
		Mode:         "weighted",
		VectorWeight: 0.5,
		BM25Weight:   0.3,
		NgramWeight:  0.2,
	}

	results := FuseScores(vectorScores, bm25Scores, ngramScores, config)
	if len(results) == 0 {
		t.Fatal("expected fused results")
	}

	// All 4 unique IDs should be present.
	if len(results) != 4 {
		t.Errorf("expected 4 fused results, got %d", len(results))
	}

	// Results should be sorted by FinalScore descending.
	for i := 1; i < len(results); i++ {
		if results[i].FinalScore > results[i-1].FinalScore {
			t.Errorf("results not sorted: [%d]=%f > [%d]=%f",
				i, results[i].FinalScore, i-1, results[i-1].FinalScore)
		}
	}
}

func TestFuseScores_RRF(t *testing.T) {
	vectorScores := map[string]float64{"a": 0.9, "b": 0.5}
	bm25Scores := map[string]float64{"b": 3.0, "a": 1.0}
	ngramScores := map[string]float64{"a": 0.7}

	config := &HybridSearchConfig{
		Mode:        "rrf",
		RRFConstant: 60,
	}

	results := FuseScores(vectorScores, bm25Scores, ngramScores, config)
	if len(results) != 2 {
		t.Fatalf("expected 2 fused results, got %d", len(results))
	}

	// "a" is rank 1 in vector (1/(60+1)), rank 2 in bm25 (1/(60+2)), rank 1 in ngram (1/(60+1))
	// "b" is rank 2 in vector (1/(60+2)), rank 1 in bm25 (1/(60+1)), no ngram score
	// "a" should have higher fused score due to appearing in all 3 retrievers
	aScore, bScore := 0.0, 0.0
	for _, r := range results {
		if r.ChunkID == "a" {
			aScore = r.FinalScore
		}
		if r.ChunkID == "b" {
			bScore = r.FinalScore
		}
	}
	if aScore <= bScore {
		t.Errorf("'a' (%f) should rank above 'b' (%f) in RRF", aScore, bScore)
	}
}

func TestFuseScores_DefaultConfig(t *testing.T) {
	config := &HybridSearchConfig{}
	config.applyDefaults()
	if config.Mode != "weighted" {
		t.Errorf("default mode should be 'weighted', got %q", config.Mode)
	}
	wV, wB, wN := config.normalizedWeights()
	sum := wV + wB + wN
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("normalized weights should sum to 1.0, got %f", sum)
	}
}

// ---------------------------------------------------------------------------
// Memory Backend HybridSearch integration
// ---------------------------------------------------------------------------

func TestMemoryBackend_HybridSearch(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	if err := backend.CreateCollection(ctx, "vs1", 3); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	chunks := []EmbeddedChunk{
		{ID: "c1", FileID: "f1", Filename: "doc1.txt", Content: "the quick brown fox jumps over the lazy dog", Embedding: []float32{0.9, 0.1, 0.0}, ChunkIndex: 0},
		{ID: "c2", FileID: "f1", Filename: "doc1.txt", Content: "machine learning and deep learning for artificial intelligence", Embedding: []float32{0.1, 0.9, 0.0}, ChunkIndex: 1},
		{ID: "c3", FileID: "f2", Filename: "doc2.txt", Content: "the quick fox is brown and fast", Embedding: []float32{0.8, 0.2, 0.0}, ChunkIndex: 0},
	}
	if err := backend.InsertChunks(ctx, "vs1", chunks); err != nil {
		t.Fatalf("InsertChunks: %v", err)
	}

	config := &HybridSearchConfig{
		Mode:         "weighted",
		VectorWeight: 0.5,
		BM25Weight:   0.3,
		NgramWeight:  0.2,
	}

	// Query embedding is close to c1/c3 (fox documents)
	queryEmbedding := []float32{0.85, 0.15, 0.0}
	results, err := backend.HybridSearch(ctx, "vs1", "quick brown fox", queryEmbedding, 10, 0, nil, config)
	if err != nil {
		t.Fatalf("HybridSearch: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected hybrid search results")
	}

	// c1 and c3 should rank above c2 (both vector and text match "quick brown fox")
	for _, r := range results {
		if r.VectorScore == nil || r.BM25Score == nil || r.NgramScore == nil {
			t.Errorf("component scores should be populated for chunk %s", r.FileID)
		}
	}

	// Top result should be c1 or c3 (both have strong vector and text signals)
	top := results[0]
	if top.Content != chunks[0].Content && top.Content != chunks[2].Content {
		t.Errorf("top result should be a fox document, got %q", top.Content)
	}
}

func TestMemoryBackend_HybridSearch_RRF(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	if err := backend.CreateCollection(ctx, "vs1", 3); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	chunks := []EmbeddedChunk{
		{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "alpha beta gamma", Embedding: []float32{1, 0, 0}, ChunkIndex: 0},
		{ID: "c2", FileID: "f1", Filename: "a.txt", Content: "delta epsilon zeta", Embedding: []float32{0, 1, 0}, ChunkIndex: 1},
	}
	if err := backend.InsertChunks(ctx, "vs1", chunks); err != nil {
		t.Fatalf("InsertChunks: %v", err)
	}

	config := &HybridSearchConfig{Mode: "rrf"}
	queryEmb := []float32{0.9, 0.1, 0.0}

	results, err := backend.HybridSearch(ctx, "vs1", "alpha beta", queryEmb, 10, 0, nil, config)
	if err != nil {
		t.Fatalf("HybridSearch RRF: %v", err)
	}

	if len(results) < 1 {
		t.Fatal("expected at least one result")
	}

	// c1 should rank first (matches both vector and text)
	if results[0].Content != "alpha beta gamma" {
		t.Errorf("expected c1 first in RRF, got %q", results[0].Content)
	}
}

func TestMemoryBackend_HybridSearch_FileFilter(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	if err := backend.CreateCollection(ctx, "vs1", 3); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	chunks := []EmbeddedChunk{
		{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "hello world", Embedding: []float32{1, 0, 0}, ChunkIndex: 0},
		{ID: "c2", FileID: "f2", Filename: "b.txt", Content: "hello world", Embedding: []float32{1, 0, 0}, ChunkIndex: 0},
	}
	if err := backend.InsertChunks(ctx, "vs1", chunks); err != nil {
		t.Fatalf("InsertChunks: %v", err)
	}

	config := &HybridSearchConfig{Mode: "weighted"}
	filter := map[string]interface{}{"file_id": "f1"}
	queryEmb := []float32{1, 0, 0}

	results, err := backend.HybridSearch(ctx, "vs1", "hello", queryEmb, 10, 0, filter, config)
	if err != nil {
		t.Fatalf("HybridSearch with filter: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result with file_id filter, got %d", len(results))
	}
	if results[0].FileID != "f1" {
		t.Errorf("expected file_id=f1, got %s", results[0].FileID)
	}
}

func TestMemoryBackend_HybridSearch_CollectionNotFound(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	config := &HybridSearchConfig{}
	_, err := backend.HybridSearch(ctx, "nonexistent", "query", []float32{1}, 10, 0, nil, config)
	if err == nil {
		t.Error("expected error for nonexistent collection")
	}
}

func TestMemoryBackend_HybridSearch_IndexRebuildOnDelete(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	if err := backend.CreateCollection(ctx, "vs1", 3); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	chunks := []EmbeddedChunk{
		{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "unique keyword alpha", Embedding: []float32{1, 0, 0}, ChunkIndex: 0},
		{ID: "c2", FileID: "f2", Filename: "b.txt", Content: "different topic beta", Embedding: []float32{0, 1, 0}, ChunkIndex: 0},
	}
	if err := backend.InsertChunks(ctx, "vs1", chunks); err != nil {
		t.Fatalf("InsertChunks: %v", err)
	}

	// Delete file f1
	if err := backend.DeleteByFileID(ctx, "vs1", "f1"); err != nil {
		t.Fatalf("DeleteByFileID: %v", err)
	}

	config := &HybridSearchConfig{Mode: "weighted"}
	queryEmb := []float32{1, 0, 0}

	results, err := backend.HybridSearch(ctx, "vs1", "unique keyword alpha", queryEmb, 10, 0, nil, config)
	if err != nil {
		t.Fatalf("HybridSearch after delete: %v", err)
	}

	for _, r := range results {
		if r.FileID == "f1" {
			t.Error("deleted file should not appear in search results")
		}
	}
}

// ---------------------------------------------------------------------------
// GenericHybridRerank (works with any VectorStoreBackend)
// ---------------------------------------------------------------------------

func TestGenericHybridRerank_Weighted(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	if err := backend.CreateCollection(ctx, "vs1", 3); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	chunks := []EmbeddedChunk{
		{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "the quick brown fox jumps over the lazy dog", Embedding: []float32{0.9, 0.1, 0.0}, ChunkIndex: 0},
		{ID: "c2", FileID: "f1", Filename: "a.txt", Content: "machine learning and deep learning", Embedding: []float32{0.1, 0.9, 0.0}, ChunkIndex: 1},
		{ID: "c3", FileID: "f2", Filename: "b.txt", Content: "the quick fox is brown and fast", Embedding: []float32{0.8, 0.2, 0.0}, ChunkIndex: 0},
	}
	if err := backend.InsertChunks(ctx, "vs1", chunks); err != nil {
		t.Fatalf("InsertChunks: %v", err)
	}

	config := &HybridSearchConfig{
		Mode:         "weighted",
		VectorWeight: 0.5,
		BM25Weight:   0.3,
		NgramWeight:  0.2,
	}

	queryEmb := []float32{0.85, 0.15, 0.0}
	results, err := GenericHybridRerank(ctx, backend, "vs1", "quick brown fox", queryEmb, 10, 0, nil, config)
	if err != nil {
		t.Fatalf("GenericHybridRerank: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("expected results from generic hybrid rerank")
	}

	// All results should have component scores populated.
	for i, r := range results {
		if r.VectorScore == nil || r.BM25Score == nil || r.NgramScore == nil {
			t.Errorf("result[%d]: component scores should be populated", i)
		}
	}

	// "quick brown fox" documents should rank above machine learning.
	top := results[0]
	if top.Content != chunks[0].Content && top.Content != chunks[2].Content {
		t.Errorf("top result should match fox query, got %q", top.Content)
	}
}

func TestGenericHybridRerank_RRF(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	if err := backend.CreateCollection(ctx, "vs1", 3); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	chunks := []EmbeddedChunk{
		{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "alpha beta gamma", Embedding: []float32{1, 0, 0}, ChunkIndex: 0},
		{ID: "c2", FileID: "f1", Filename: "a.txt", Content: "delta epsilon zeta", Embedding: []float32{0, 1, 0}, ChunkIndex: 1},
	}
	if err := backend.InsertChunks(ctx, "vs1", chunks); err != nil {
		t.Fatalf("InsertChunks: %v", err)
	}

	config := &HybridSearchConfig{Mode: "rrf"}
	queryEmb := []float32{0.9, 0.1, 0.0}

	results, err := GenericHybridRerank(ctx, backend, "vs1", "alpha beta", queryEmb, 10, 0, nil, config)
	if err != nil {
		t.Fatalf("GenericHybridRerank RRF: %v", err)
	}

	if len(results) < 1 {
		t.Fatal("expected at least one result")
	}

	// c1 matches both vector and text signals.
	if results[0].Content != "alpha beta gamma" {
		t.Errorf("expected c1 first, got %q", results[0].Content)
	}
}

func TestGenericHybridRerank_Threshold(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	if err := backend.CreateCollection(ctx, "vs1", 3); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	chunks := []EmbeddedChunk{
		{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "hello world", Embedding: []float32{1, 0, 0}, ChunkIndex: 0},
		{ID: "c2", FileID: "f1", Filename: "a.txt", Content: "goodbye moon", Embedding: []float32{0, 0.01, 0}, ChunkIndex: 1},
	}
	if err := backend.InsertChunks(ctx, "vs1", chunks); err != nil {
		t.Fatalf("InsertChunks: %v", err)
	}

	config := &HybridSearchConfig{Mode: "weighted"}
	queryEmb := []float32{1, 0, 0}

	// High threshold should filter out the weakly matching c2.
	results, err := GenericHybridRerank(ctx, backend, "vs1", "hello", queryEmb, 10, 0.5, nil, config)
	if err != nil {
		t.Fatalf("GenericHybridRerank threshold: %v", err)
	}

	for _, r := range results {
		if r.Score < 0.5 {
			t.Errorf("result with score %f should have been filtered by threshold 0.5", r.Score)
		}
	}
}

func TestGenericHybridRerank_TopK(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	if err := backend.CreateCollection(ctx, "vs1", 3); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	chunks := []EmbeddedChunk{
		{ID: "c1", FileID: "f1", Filename: "a.txt", Content: "alpha one", Embedding: []float32{0.9, 0.1, 0.0}, ChunkIndex: 0},
		{ID: "c2", FileID: "f1", Filename: "a.txt", Content: "alpha two", Embedding: []float32{0.8, 0.2, 0.0}, ChunkIndex: 1},
		{ID: "c3", FileID: "f1", Filename: "a.txt", Content: "alpha three", Embedding: []float32{0.7, 0.3, 0.0}, ChunkIndex: 2},
	}
	if err := backend.InsertChunks(ctx, "vs1", chunks); err != nil {
		t.Fatalf("InsertChunks: %v", err)
	}

	config := &HybridSearchConfig{Mode: "weighted"}
	queryEmb := []float32{0.85, 0.15, 0.0}

	results, err := GenericHybridRerank(ctx, backend, "vs1", "alpha", queryEmb, 2, 0, nil, config)
	if err != nil {
		t.Fatalf("GenericHybridRerank topK: %v", err)
	}

	if len(results) > 2 {
		t.Errorf("topK=2 but got %d results", len(results))
	}
}

func TestGenericHybridRerank_EmptyCollection(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	if err := backend.CreateCollection(ctx, "vs1", 3); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	config := &HybridSearchConfig{Mode: "weighted"}
	queryEmb := []float32{1, 0, 0}

	results, err := GenericHybridRerank(ctx, backend, "vs1", "anything", queryEmb, 10, 0, nil, config)
	if err != nil {
		t.Fatalf("GenericHybridRerank empty: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("expected 0 results for empty collection, got %d", len(results))
	}
}

func TestGenericHybridRerank_CollectionNotFound(t *testing.T) {
	backend := NewMemoryBackend(MemoryBackendConfig{})
	ctx := context.Background()

	config := &HybridSearchConfig{}
	_, err := GenericHybridRerank(ctx, backend, "nonexistent", "query", []float32{1}, 10, 0, nil, config)
	if err == nil {
		t.Error("expected error for nonexistent collection")
	}
}
