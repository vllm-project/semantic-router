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

import "sort"

// MergeSearchResults keeps each chunk's best score across the searches of one
// query. A long query is searched once per window of itself, so the same chunk
// comes back from several of them, and the best score is the aggregation that
// ranks best over passages of a long input.
//
// Chunks are identified by file, chunk index and content. Not every backend
// reports a chunk index: llama_stack leaves every hit at zero, and its hybrid
// hits carry no file id either, so file and index alone would collapse a whole
// response into its best scoring row. Content joins the key because it only
// ever separates hits, so the same chunk found through several windows still
// merges, while two chunks that share text stay apart on file and index. Equal
// scores order by file, chunk index and content, which is the whole key, so the
// same batches always produce the same page even when the rows they tie on are
// the ones a chunk index would have separated.
func MergeSearchResults(topK int, batches ...[]SearchResult) []SearchResult {
	type chunkKey struct {
		fileID  string
		index   int
		content string
	}

	best := make(map[chunkKey]SearchResult)
	for _, batch := range batches {
		for _, result := range batch {
			key := chunkKey{fileID: result.FileID, index: result.ChunkIndex, content: result.Content}
			if current, seen := best[key]; !seen || result.Score > current.Score {
				best[key] = result
			}
		}
	}

	merged := make([]SearchResult, 0, len(best))
	for _, result := range best {
		merged = append(merged, result)
	}
	sort.Slice(merged, func(i, j int) bool {
		if merged[i].Score != merged[j].Score {
			return merged[i].Score > merged[j].Score
		}
		if merged[i].FileID != merged[j].FileID {
			return merged[i].FileID < merged[j].FileID
		}
		if merged[i].ChunkIndex != merged[j].ChunkIndex {
			return merged[i].ChunkIndex < merged[j].ChunkIndex
		}
		return merged[i].Content < merged[j].Content
	})
	if topK > 0 && len(merged) > topK {
		merged = merged[:topK]
	}
	return merged
}
