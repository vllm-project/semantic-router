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

import (
	"context"
	"strings"
	"testing"
)

type fakeClient struct {
	response any
	err      error
}

func (f *fakeClient) CustomCommand(_ context.Context, _ []string) (any, error) {
	return f.response, f.err
}

func mod(name string, ver int64) map[string]any {
	return map[string]any{"name": name, "ver": ver}
}

func TestEnsureSearchModuleVersion(t *testing.T) {
	cases := []struct {
		name        string
		response    any
		respErr     error
		wantErr     bool
		errContains []string
	}{
		{name: "1.2.0 passes", response: []any{mod("search", 0x010200)}},
		{name: "1.2.1 passes", response: []any{mod("search", 0x010201)}},
		{name: "2.0.0 passes", response: []any{mod("search", 0x020000)}},
		{name: "search alongside other modules", response: []any{mod("bf", 0x010001), mod("search", 0x010200), mod("json", 0x010002)}},
		{
			name: "1.0.0 rejected", response: []any{mod("search", 0x010000)},
			wantErr: true, errContains: []string{"1.0.0", "1.2.0"},
		},
		{
			name: "1.1.0 rejected", response: []any{mod("search", 0x010100)},
			wantErr: true, errContains: []string{"1.1.0", "1.2.0"},
		},
		{
			name: "no search module", response: []any{mod("bf", 0x010001), mod("json", 0x010002)},
			wantErr: true, errContains: []string{"not loaded", "1.2.0"},
		},
		{
			name: "empty list", response: []any{},
			wantErr: true, errContains: []string{"not loaded"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := &fakeClient{response: tc.response, err: tc.respErr}
			err := EnsureSearchModuleVersion(context.Background(), client, SearchModuleMinVersion)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				for _, sub := range tc.errContains {
					if !strings.Contains(err.Error(), sub) {
						t.Errorf("error %q missing %q", err.Error(), sub)
					}
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
