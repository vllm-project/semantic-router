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

package selection

import (
	"context"
	"strings"
	"testing"
)

// sharedPrefix is longer than the 32-byte window the old hashQuery truncated to.
const sharedPrefix = "You are a helpful assistant. Answer the user request: "

func TestRouterDCSelector_HashQueryDistinguishesSharedPrefix(t *testing.T) {
	selector := NewRouterDCSelector(DefaultRouterDCConfig())

	poem := sharedPrefix + "write a poem about the ocean"
	sql := sharedPrefix + "SQL to drop the production users table"

	poemHash := selector.hashQuery(poem)
	sqlHash := selector.hashQuery(sql)

	if poemHash == sqlHash {
		t.Fatalf("queries sharing a %d-byte prefix hashed to the same key %q", len(sharedPrefix), poemHash)
	}
	if again := selector.hashQuery(poem); again != poemHash {
		t.Fatalf("hashQuery is not stable: %q != %q", again, poemHash)
	}
	if len(poemHash) != 64 || strings.Trim(poemHash, "0123456789abcdef") != "" {
		t.Fatalf("expected 64-char lowercase hex SHA-256 digest, got %q", poemHash)
	}
}

func TestRouterDCSelector_UpdateFeedbackKeysAffinityPerQuery(t *testing.T) {
	selector := NewRouterDCSelector(DefaultRouterDCConfig())
	ctx := context.Background()

	poem := sharedPrefix + "write a poem about the ocean"
	sql := sharedPrefix + "SQL to drop the production users table"

	if err := selector.UpdateFeedback(ctx, &Feedback{Query: poem, WinnerModel: "poet"}); err != nil {
		t.Fatalf("UpdateFeedback(poem) error = %v", err)
	}
	if err := selector.UpdateFeedback(ctx, &Feedback{Query: sql, WinnerModel: "dba"}); err != nil {
		t.Fatalf("UpdateFeedback(sql) error = %v", err)
	}

	selector.affinityMu.RLock()
	defer selector.affinityMu.RUnlock()

	if got := len(selector.affinityMatrix); got != 2 {
		t.Fatalf("expected 2 affinity buckets, got %d: %v", got, selector.affinityMatrix)
	}

	poemBucket := selector.affinityMatrix[selector.hashQuery(poem)]
	sqlBucket := selector.affinityMatrix[selector.hashQuery(sql)]

	if poemBucket["poet"] == 0 || poemBucket["dba"] != 0 {
		t.Fatalf("poem bucket should only hold poet affinity, got %v", poemBucket)
	}
	if sqlBucket["dba"] == 0 || sqlBucket["poet"] != 0 {
		t.Fatalf("sql bucket should only hold dba affinity, got %v", sqlBucket)
	}
}
