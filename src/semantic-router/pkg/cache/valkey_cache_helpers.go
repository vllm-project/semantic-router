package cache

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// searchMatch holds the parsed fields from a vector search result.
type searchMatch struct {
	distance     float64
	responseBody interface{}
}

// parseBestMatch extracts the best-match distance and response body from a Valkey FT.SEARCH vector result.
// Returns nil when no valid match is found.
func parseBestMatch(searchResult interface{}) *searchMatch {
	resultsArray, ok := searchResult.([]interface{})
	if !ok || len(resultsArray) < 2 {
		return nil
	}

	totalResults, ok := resultsArray[0].(int64)
	if !ok || totalResults == 0 {
		return nil
	}

	docMap, ok := resultsArray[1].(map[string]interface{})
	if !ok {
		logging.Debugf("ValkeyCache.FindSimilarWithThreshold: invalid result format, expected map at index 1, got %T", resultsArray[1])
		return nil
	}

	// FT.SEARCH returns results ordered by distance, but Go map iteration
	// in the valkey-glide response loses that ordering. Iterate all docs and
	// pick the best one.
	var best *searchMatch
	for _, docValue := range docMap {
		fieldsMap, mapOk := docValue.(map[string]interface{})
		if !mapOk {
			logging.Debugf("ValkeyCache.FindSimilarWithThreshold: invalid fields format, expected map, got %T", docValue)
			continue
		}

		distanceVal, exists := fieldsMap["vector_distance"]
		if !exists {
			continue
		}

		var distance float64
		if _, err := fmt.Sscanf(fmt.Sprint(distanceVal), "%f", &distance); err != nil {
			logging.Debugf("ValkeyCache.FindSimilarWithThreshold: failed to parse distance value: %v", err)
			continue
		}

		if best == nil || distance < best.distance {
			best = &searchMatch{
				distance:     distance,
				responseBody: fieldsMap["response_body"],
			}
		}
	}

	return best
}

// escapeTagValue escapes punctuation and whitespace in a string so it can be
// safely used inside a Valkey/Redis TAG query expression (@field:{value}).
// TAG queries treat punctuation characters as token separators; backslash-
// escaping them preserves the literal value.
// Reference: https://forum.redis.com/t/tag-fields-and-escaping/96
func escapeTagValue(s string) string {
	// Characters that must be backslash-escaped in TAG query values.
	// These are the punctuation/space characters that the RediSearch/Valkey
	// query tokenizer treats as separators.
	const specialChars = " \t,.<>{}[]\"':;!@#$%^&*()-+=~|/\\"

	var b strings.Builder
	b.Grow(len(s) + 8) // most IDs need only a few extra backslashes
	for _, c := range s {
		if strings.ContainsRune(specialChars, c) {
			b.WriteByte('\\')
		}
		b.WriteRune(c)
	}
	return b.String()
}

// distanceToSimilarity converts a vector distance to a similarity score based on the metric type.
func distanceToSimilarity(metricType string, distance float64) float32 {
	switch metricType {
	case "COSINE":
		return 1.0 - float32(distance)/2.0
	case "IP":
		return float32(distance)
	case "L2":
		return 1.0 / (1.0 + float32(distance))
	default:
		return 1.0 - float32(distance)
	}
}

// extractResponseBody returns the response bytes from a search match, or nil if missing/empty.
func extractResponseBody(match *searchMatch) []byte {
	if match.responseBody == nil {
		return nil
	}
	s := fmt.Sprint(match.responseBody)
	if s == "" {
		return nil
	}
	return []byte(s)
}
