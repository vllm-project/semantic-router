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
	"testing"
	"time"
)

func TestResolveAutoSaveInterval(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want time.Duration
	}{
		{name: "empty falls back to default", raw: "", want: defaultAutoSaveInterval},
		{name: "valid value honored", raw: "45s", want: 45 * time.Second},
		{name: "zero falls back to default", raw: "0s", want: defaultAutoSaveInterval},
		{name: "negative falls back to default", raw: "-5s", want: defaultAutoSaveInterval},
		{name: "malformed falls back to default", raw: "banana", want: defaultAutoSaveInterval},
		{name: "above max falls back to default", raw: "48h", want: defaultAutoSaveInterval},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveAutoSaveInterval(tt.raw)
			if got != tt.want {
				t.Fatalf("resolveAutoSaveInterval(%q) = %s, want %s", tt.raw, got, tt.want)
			}
			if got <= 0 {
				t.Fatalf("resolveAutoSaveInterval(%q) returned non-positive %s", tt.raw, got)
			}
		})
	}
}
