//go:build !windows && cgo

package cache

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseBestMatch_SelectsSmallestDistance(t *testing.T) {
	// Simulate a FT.SEARCH result with 3 documents.
	// Go map iteration order is random, so the function must
	// always return the doc with the smallest distance.
	searchResult := []interface{}{
		int64(3),
		map[string]interface{}{
			"cache:doc1": map[string]interface{}{
				"vector_distance": "0.50",
				"response_body":   "resp_worst",
			},
			"cache:doc2": map[string]interface{}{
				"vector_distance": "0.05",
				"response_body":   "resp_best",
			},
			"cache:doc3": map[string]interface{}{
				"vector_distance": "0.30",
				"response_body":   "resp_mid",
			},
		},
	}

	// Run multiple times to defeat random map iteration order.
	for i := 0; i < 50; i++ {
		match := parseBestMatch(searchResult)
		assert.NotNil(t, match)
		assert.InDelta(t, 0.05, match.distance, 1e-9)
		assert.Equal(t, "resp_best", match.responseBody)
	}
}

func TestParseBestMatch_SingleDoc(t *testing.T) {
	searchResult := []interface{}{
		int64(1),
		map[string]interface{}{
			"cache:only": map[string]interface{}{
				"vector_distance": "0.12",
				"response_body":   "only_resp",
			},
		},
	}

	match := parseBestMatch(searchResult)
	assert.NotNil(t, match)
	assert.InDelta(t, 0.12, match.distance, 1e-9)
	assert.Equal(t, "only_resp", match.responseBody)
}

func TestParseBestMatch_EmptyResults(t *testing.T) {
	assert.Nil(t, parseBestMatch(nil))
	assert.Nil(t, parseBestMatch([]interface{}{int64(0)}))
	assert.Nil(t, parseBestMatch([]interface{}{}))
}

func TestParseBestMatch_SkipsInvalidDocs(t *testing.T) {
	searchResult := []interface{}{
		int64(2),
		map[string]interface{}{
			"cache:bad": "not_a_map",
			"cache:good": map[string]interface{}{
				"vector_distance": "0.10",
				"response_body":   "good_resp",
			},
		},
	}

	match := parseBestMatch(searchResult)
	assert.NotNil(t, match)
	assert.InDelta(t, 0.10, match.distance, 1e-9)
	assert.Equal(t, "good_resp", match.responseBody)
}
