// Copyright 2025 The vLLM Semantic Router Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cache

type EvictionPolicy interface {
	SelectVictim(entries []CacheEntry) int
}

type FIFOPolicy struct{}

func (p *FIFOPolicy) SelectVictim(entries []CacheEntry) int {
	if len(entries) == 0 {
		return -1
	}

	oldestIdx := 0
	for i := 1; i < len(entries); i++ {
		if entries[i].Timestamp.Before(entries[oldestIdx].Timestamp) {
			oldestIdx = i
		}
	}
	return oldestIdx
}

type LRUPolicy struct{}

func (p *LRUPolicy) SelectVictim(entries []CacheEntry) int {
	if len(entries) == 0 {
		return -1
	}

	oldestIdx := 0
	for i := 1; i < len(entries); i++ {
		if entries[i].LastAccessAt.Before(entries[oldestIdx].LastAccessAt) {
			oldestIdx = i
		}
	}
	return oldestIdx
}

type LFUPolicy struct{}

func (p *LFUPolicy) SelectVictim(entries []CacheEntry) int {
	if len(entries) == 0 {
		return -1
	}

	victimIdx := 0
	for i := 1; i < len(entries); i++ {
		if entries[i].HitCount < entries[victimIdx].HitCount {
			victimIdx = i
		} else if entries[i].HitCount == entries[victimIdx].HitCount {
			// Use LRU as tiebreaker to avoid random selection
			if entries[i].LastAccessAt.Before(entries[victimIdx].LastAccessAt) {
				victimIdx = i
			}
		}
	}
	return victimIdx
}
