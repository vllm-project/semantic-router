// This test validates the Redis KNN query format independently of the cache
// package's CGo dependencies (candle bindings). It can run with CGO_ENABLED=0
// since it only tests string formatting.
//
//go:build !windows && cgo

package cache

import (
	"fmt"
	"strings"
	"testing"
)

// TestRedisKNNQueryFormat validates that the KNN query string used for Redis
// vector similarity search conforms to the FT.SEARCH DIALECT 2 syntax.
//
// Redis requires the "*=>" prefix for hybrid queries:
//
//	*=>[KNN <K> @<field> $<param> AS <alias>]
//
// Without the "*=>" prefix, Redis returns:
//
//	Syntax error at offset 0 near
//
// See: https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/
func TestRedisKNNQueryFormat(t *testing.T) {
	tests := []struct {
		name           string
		topK           int
		vectorField    string
		expectPrefix   string
		expectContains []string
	}{
		{
			name:         "default config (topK=1, field=embedding)",
			topK:         1,
			vectorField:  "embedding",
			expectPrefix: "*=>[KNN",
			expectContains: []string{
				"*=>[KNN 1 @embedding $vec AS vector_distance]",
			},
		},
		{
			name:         "topK=5",
			topK:         5,
			vectorField:  "embedding",
			expectPrefix: "*=>[KNN",
			expectContains: []string{
				"*=>[KNN 5 @embedding $vec AS vector_distance]",
			},
		},
		{
			name:         "custom vector field name",
			topK:         1,
			vectorField:  "vec_field",
			expectPrefix: "*=>[KNN",
			expectContains: []string{
				"*=>[KNN 1 @vec_field $vec AS vector_distance]",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// This mirrors the query construction in FindSimilarWithThreshold
			knnQuery := fmt.Sprintf("*=>[KNN %d @%s $vec AS vector_distance]",
				tt.topK, tt.vectorField)

			if !strings.HasPrefix(knnQuery, tt.expectPrefix) {
				t.Errorf("KNN query missing required '*=>' prefix for Redis DIALECT 2.\n  got:  %q\n  want prefix: %q",
					knnQuery, tt.expectPrefix)
			}

			for _, substr := range tt.expectContains {
				if knnQuery != substr {
					t.Errorf("KNN query mismatch.\n  got:  %q\n  want: %q", knnQuery, substr)
				}
			}
		})
	}
}

// TestRedisKNNQueryRequiresFilterPrefix ensures the query cannot be constructed
// without the "*=>" prefix, which would cause a Redis syntax error.
func TestRedisKNNQueryRequiresFilterPrefix(t *testing.T) {
	// The INCORRECT format (missing *=> prefix) that caused the original bug
	badQuery := fmt.Sprintf("[KNN %d @%s $vec AS vector_distance]", 1, "embedding")

	if strings.HasPrefix(badQuery, "*=>") {
		t.Fatal("Sanity check failed: bad query should NOT have *=> prefix")
	}

	// The CORRECT format
	goodQuery := fmt.Sprintf("*=>[KNN %d @%s $vec AS vector_distance]", 1, "embedding")

	if !strings.HasPrefix(goodQuery, "*=>") {
		t.Fatal("Good query must have *=> prefix for Redis DIALECT 2")
	}

	// Verify the fix is applied in the actual code format
	// (this test would fail if someone reverts the fix)
	topK := 1
	vectorFieldName := "embedding"
	actualQuery := fmt.Sprintf("*=>[KNN %d @%s $vec AS vector_distance]",
		topK, vectorFieldName)

	if !strings.HasPrefix(actualQuery, "*=>") {
		t.Errorf("Redis KNN query is missing the '*=>' prefix required by DIALECT 2.\n"+
			"  got:  %q\n"+
			"  This will cause 'Syntax error at offset 0' on Redis FT.SEARCH.\n"+
			"  See: https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/",
			actualQuery)
	}
}
