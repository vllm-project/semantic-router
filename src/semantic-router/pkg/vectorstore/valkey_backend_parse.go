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
	"encoding/binary"
	"math"
	"sort"
	"strconv"
	"strings"
)

// float32SliceToBytes converts a float32 slice to a little-endian byte slice
// suitable for Valkey vector storage.
func float32SliceToBytes(floats []float32) []byte {
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// escapeTagValue escapes special characters in a Valkey tag filter value.
// Tag values containing hyphens, dots, or other special chars need escaping.
func escapeTagValue(val string) string {
	replacer := strings.NewReplacer(
		"-", "\\-",
		".", "\\.",
		":", "\\:",
		"/", "\\/",
		" ", "\\ ",
	)
	return replacer.Replace(val)
}

// extractKeysFromSearchResult extracts document keys from an FT.SEARCH NOCONTENT result.
// valkey-glide v2 returns NOCONTENT results as: [totalCount, "key1", "key2", ...]
// but may also return map-based results: [totalCount, map[key: map[...]], ...].
// This function handles both formats.
func extractKeysFromSearchResult(result any) []string {
	arr, ok := result.([]interface{})
	if !ok || len(arr) < 2 {
		return nil
	}

	var keys []string
	for i := 1; i < len(arr); i++ {
		switch v := arr[i].(type) {
		case string:
			// NOCONTENT flat format: element is the key string directly.
			keys = append(keys, v)
		case map[string]interface{}:
			// Map format: key is the map key.
			for docKey := range v {
				keys = append(keys, docKey)
			}
		}
	}
	return keys
}

// parseSearchResults parses the FT.SEARCH result into SearchResult slice.
// valkey-glide v2 CustomCommand returns results as:
//
//	[totalCount, map[docKey1: map[fields...], docKey2: map[fields...], ...]]
//
// All documents are returned in a single map (the second element).
// This follows the same pattern as the Valkey cache backend (PR #1540).
func (v *ValkeyBackend) parseSearchResults(result any, threshold float32) ([]SearchResult, error) {
	arr, ok := result.([]interface{})
	if !ok || len(arr) < 1 {
		return nil, nil
	}

	totalCount, ok := toInt64(arr[0])
	if !ok || totalCount <= 0 {
		return nil, nil
	}

	var results []SearchResult

	// valkey-glide may return docs as a single map or multiple map entries.
	for i := 1; i < len(arr); i++ {
		docMap, ok := arr[i].(map[string]interface{})
		if !ok {
			continue
		}

		// Each key in docMap is a document key, value is the fields map.
		for _, docValue := range docMap {
			fieldsMap, mapOk := docValue.(map[string]interface{})
			if !mapOk {
				continue
			}

			result, ok := v.parseSearchResultFields(fieldsMap, threshold)
			if ok {
				results = append(results, result)
			}
		}
	}

	// Sort by score descending so callers always get best matches first.
	// FT.SEARCH returns results ordered by distance, but Go map iteration
	// in the valkey-glide response loses that ordering.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

func (v *ValkeyBackend) parseSearchResultFields(
	fields map[string]interface{},
	threshold float32,
) (SearchResult, bool) {
	score, ok := parseScoreFromMap(fields, "vector_distance", v.metricType)
	if !ok || score < float64(threshold) {
		return SearchResult{}, false
	}

	chunkIndex, ok := parseValkeyInt(fields["chunk_index"])
	if !ok {
		return SearchResult{}, false
	}
	fileID, ok := valkeyStringField(fields, "file_id")
	if !ok {
		return SearchResult{}, false
	}
	filename, ok := valkeyStringField(fields, "filename")
	if !ok {
		return SearchResult{}, false
	}
	content, ok := valkeyStringField(fields, "content")
	if !ok {
		return SearchResult{}, false
	}

	return SearchResult{
		FileID:     fileID,
		Filename:   filename,
		Content:    content,
		Score:      score,
		ChunkIndex: chunkIndex,
	}, true
}

func valkeyStringField(fields map[string]interface{}, key string) (string, bool) {
	raw, exists := fields[key]
	if !exists || raw == nil {
		return "", false
	}
	switch val := raw.(type) {
	case string:
		return val, true
	case []byte:
		return string(val), true
	default:
		return "", false
	}
}

// parseScoreFromMap extracts a distance value from the fields map and converts it to similarity.
func parseScoreFromMap(fields map[string]interface{}, key string, metricType string) (float64, bool) {
	raw, exists := fields[key]
	if !exists {
		return 0, false
	}
	distance, ok := parseValkeyFloat(raw)
	if !ok {
		return 0, false
	}
	score := distanceToSimilarity(metricType, distance)
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return 0, false
	}
	return score, true
}

func parseValkeyFloat(raw interface{}) (float64, bool) {
	switch val := raw.(type) {
	case float64:
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return 0, false
		}
		return val, true
	case float32:
		f := float64(val)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return 0, false
		}
		return f, true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	case string:
		return parseFiniteFloatString(val)
	case []byte:
		return parseFiniteFloatString(string(val))
	default:
		return 0, false
	}
}

func parseFiniteFloatString(value string) (float64, bool) {
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
		return 0, false
	}
	return parsed, true
}

func parseValkeyInt(raw interface{}) (int, bool) {
	value, ok := toInt64(raw)
	if !ok {
		return 0, false
	}
	return int(value), true
}

// distanceToSimilarity converts a vector distance to a similarity score based on the metric type.
// Follows the same conversion as the Valkey cache backend (PR #1540).
func distanceToSimilarity(metricType string, distance float64) float64 {
	switch strings.ToUpper(metricType) {
	case "COSINE":
		return 1.0 - distance/2.0
	case "L2":
		return 1.0 / (1.0 + distance)
	case "IP":
		return distance
	default:
		return 1.0 - distance
	}
}

// toInt64 converts an interface{} to int64, handling both int64 and string representations.
func toInt64(v interface{}) (int64, bool) {
	switch val := v.(type) {
	case int:
		return int64(val), true
	case int64:
		return val, true
	case float64:
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return 0, false
		}
		return int64(val), true
	case string:
		n, err := strconv.ParseInt(val, 10, 64)
		return n, err == nil
	case []byte:
		n, err := strconv.ParseInt(string(val), 10, 64)
		return n, err == nil
	default:
		return 0, false
	}
}
