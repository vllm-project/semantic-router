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

package valkey

import "strings"

// DistanceToSimilarity converts the distance yielded by an FT.SEARCH KNN query
// into a similarity score, so that callers can compare it against a
// larger-is-more-similar threshold.
//
// Every branch must invert the ordering. valkey-search reports all three
// metrics as distances - "the KNN search algorithm operates to locate vectors
// that are the nearest to the query vector, i.e., looking for the smallest
// distance value", and "the computation of the distance metrics is adjusted
// from their classical definitions in order to possess this property"
// (https://valkey.io/commands/ft.create/). Inner product and cosine are
// classically similarities, so valkey-search negates them; converting back is
// what this function does. RediSearch defines the three metrics identically
// (https://redis.io/docs/latest/develop/ai/search-and-query/vectors/), so the
// Redis-backed cache shares this conversion:
//
//	COSINE: d = 1 - cos(u, v)      bounded by 2
//	IP:     d = 1 - u*v
//	L2:     d = ||u - v||          unbounded
//
// L2 is unbounded, so it is squashed rather than subtracted. An unknown metric
// falls back to 1-d; unlike the per-metric branches this is only a guess, but
// the index builders reject or warn about unknown metrics before any search
// runs, so it should be unreachable.
func DistanceToSimilarity(metricType string, distance float64) float64 {
	switch strings.ToUpper(metricType) {
	case "COSINE":
		return 1.0 - distance/2.0
	case "IP":
		return 1.0 - distance
	case "L2":
		return 1.0 / (1.0 + distance)
	default:
		return 1.0 - distance
	}
}
