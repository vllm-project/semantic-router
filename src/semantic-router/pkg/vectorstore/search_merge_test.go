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

import "testing"

func TestMergeSearchResultsKeepsTheBestScorePerChunk(t *testing.T) {
	first := []SearchResult{
		{FileID: "file_a", Filename: "a.txt", ChunkIndex: 0, Content: "alpha", Score: 0.30},
		{FileID: "file_b", Filename: "b.txt", ChunkIndex: 0, Content: "beta", Score: 0.90},
	}
	second := []SearchResult{
		{FileID: "file_a", Filename: "a.txt", ChunkIndex: 0, Content: "alpha", Score: 0.80},
		{FileID: "file_a", Filename: "a.txt", ChunkIndex: 1, Content: "gamma", Score: 0.40},
	}

	merged := MergeSearchResults(0, first, second)
	if len(merged) != 3 {
		t.Fatalf("merged %d chunks, want 3", len(merged))
	}
	want := []struct {
		fileID string
		chunk  int
		score  float64
	}{{"file_b", 0, 0.90}, {"file_a", 0, 0.80}, {"file_a", 1, 0.40}}
	for i, expected := range want {
		got := merged[i]
		if got.FileID != expected.fileID || got.ChunkIndex != expected.chunk || got.Score != expected.score {
			t.Fatalf("rank %d is %s#%d at %.2f, want %s#%d at %.2f",
				i, got.FileID, got.ChunkIndex, got.Score, expected.fileID, expected.chunk, expected.score)
		}
	}
	if merged[1].Filename != "a.txt" || merged[1].Content != "alpha" {
		t.Errorf("merge dropped the row it kept: %+v", merged[1])
	}
}

func TestMergeSearchResultsSeparatesChunksThatShareContent(t *testing.T) {
	batch := []SearchResult{
		{FileID: "file_a", ChunkIndex: 0, Content: "same text", Score: 0.50},
		{FileID: "file_b", ChunkIndex: 0, Content: "same text", Score: 0.40},
	}

	if merged := MergeSearchResults(0, batch); len(merged) != 2 {
		t.Fatalf("merged %d chunks, want both files kept", len(merged))
	}
}

func TestMergeSearchResultsAppliesTopK(t *testing.T) {
	batch := []SearchResult{
		{FileID: "file_a", ChunkIndex: 0, Score: 0.10},
		{FileID: "file_a", ChunkIndex: 1, Score: 0.20},
		{FileID: "file_a", ChunkIndex: 2, Score: 0.30},
	}

	merged := MergeSearchResults(2, batch)
	if len(merged) != 2 || merged[0].ChunkIndex != 2 || merged[1].ChunkIndex != 1 {
		t.Fatalf("top 2 is %+v", merged)
	}
}

func TestMergeSearchResultsOrdersTiesDeterministically(t *testing.T) {
	batch := []SearchResult{
		{FileID: "file_b", ChunkIndex: 1, Score: 0.50},
		{FileID: "file_a", ChunkIndex: 2, Score: 0.50},
		{FileID: "file_a", ChunkIndex: 1, Score: 0.50},
	}

	for i := 0; i < 5; i++ {
		merged := MergeSearchResults(0, batch)
		if merged[0].FileID != "file_a" || merged[0].ChunkIndex != 1 ||
			merged[1].FileID != "file_a" || merged[1].ChunkIndex != 2 ||
			merged[2].FileID != "file_b" {
			t.Fatalf("tie order changed on run %d: %+v", i, merged)
		}
	}
}

func TestMergeSearchResultsHandlesNoBatches(t *testing.T) {
	if merged := MergeSearchResults(5); len(merged) != 0 {
		t.Fatalf("merging nothing returned %+v", merged)
	}
}

// Llama Stack reports no chunk index, so every hit of one file arrives at index
// zero. A single query vector goes through the merge too, which is where a file
// would otherwise lose every chunk but its best scoring one.
func TestMergeSearchResultsKeepsChunksOfOneFileWithoutAChunkIndex(t *testing.T) {
	response := []byte(`{"data": [
		{"content": [{"type": "text", "text": "refunds take 14 business days"}], "file_id": "file_001", "filename": "handbook.txt", "score": 0.95},
		{"content": [{"type": "text", "text": "status updates every 30 minutes"}], "file_id": "file_001", "filename": "handbook.txt", "score": 0.82}
	]}`)

	parsed, err := parseLlamaStackSearchResults(response, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(parsed) != 2 {
		t.Fatalf("parsed %d hits of one file, want 2", len(parsed))
	}

	merged := MergeSearchResults(10, parsed)
	if len(merged) != 2 {
		t.Fatalf("merging one batch of 2 chunks returned %d, want both: %+v", len(merged), merged)
	}
	if merged[0].Content != "refunds take 14 business days" || merged[1].Content != "status updates every 30 minutes" {
		t.Fatalf("merge lost a chunk of the file: %+v", merged)
	}
}

// Two chunks of one Llama Stack file share a file id and a zero chunk index, so
// they can also tie on score. The merge collects its rows from a map, which has
// no order, and sort.Slice is not stable, so a comparator that stops at the
// chunk index lets either row take the last top K place.
func TestMergeSearchResultsOrdersTiedScoresDeterministically(t *testing.T) {
	batch := []SearchResult{
		{FileID: "file_001", Filename: "handbook.txt", ChunkIndex: 0, Content: "refunds take 14 business days", Score: 0.5},
		{FileID: "file_001", Filename: "handbook.txt", ChunkIndex: 0, Content: "status updates every 30 minutes", Score: 0.5},
	}

	for run := 0; run < 500; run++ {
		top := MergeSearchResults(1, batch)
		if len(top) != 1 {
			t.Fatalf("top 1 of a tied pair returned %d rows", len(top))
		}
		if top[0].Content != "refunds take 14 business days" {
			t.Fatalf("run %d cut the tied pair down to %q", run, top[0].Content)
		}

		all := MergeSearchResults(0, batch)
		if len(all) != 2 || all[0].Content != "refunds take 14 business days" || all[1].Content != "status updates every 30 minutes" {
			t.Fatalf("run %d ordered the tied pair as %+v", run, all)
		}
	}
}

// Hybrid Llama Stack hits carry no file id either, so they would all share the
// same empty key.
func TestMergeSearchResultsKeepsHitsThatReportNoFile(t *testing.T) {
	response := []byte(`{"data": [
		{"content": [{"type": "text", "text": "rrf result 1"}], "score": 0.039},
		{"content": [{"type": "text", "text": "rrf result 2"}], "score": 0.028},
		{"content": [{"type": "text", "text": "rrf result 3"}], "score": 0.010}
	]}`)

	parsed, err := parseLlamaStackSearchResults(response, 0, false)
	if err != nil {
		t.Fatal(err)
	}

	if merged := MergeSearchResults(10, parsed); len(merged) != len(parsed) {
		t.Fatalf("merged %d of %d hits that report no file: %+v", len(merged), len(parsed), merged)
	}
}
