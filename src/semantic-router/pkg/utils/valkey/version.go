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

// Package valkey contains shared helpers for Valkey-backed components
// (vector store, semantic cache, agentic memory).
package valkey

import (
	"context"
	"fmt"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// SearchModuleMinVersion is the minimum valkey-search version required for the
// TEXT field type used in our FT.CREATE schemas.
const SearchModuleMinVersion uint32 = (1 << 16) | (2 << 8) // 1.2.0 == 0x010200

// ModuleLister is the minimal subset of *glide.Client we depend on, abstracted
// for testability.
type ModuleLister interface {
	CustomCommand(ctx context.Context, args []string) (any, error)
}

// EnsureSearchModuleVersion runs MODULE LIST and returns an error if the loaded
// `search` module is missing or has a version below minPacked.
func EnsureSearchModuleVersion(ctx context.Context, client ModuleLister, minPacked uint32) error {
	raw, err := client.CustomCommand(ctx, []string{"MODULE", "LIST"})
	if err != nil {
		logging.Warnf("valkey: MODULE LIST failed (%v), skipping search module version pre-check", err)
		return nil
	}
	arr, ok := raw.([]any)
	if !ok {
		logging.Warnf("valkey: MODULE LIST returned unexpected type %T, skipping pre-check", raw)
		return nil
	}

	for _, entry := range arr {
		m, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		if name, _ := m["name"].(string); name != "search" {
			continue
		}
		verRaw, ok := m["ver"].(int64)
		if !ok || verRaw < 0 || verRaw > math.MaxUint32 {
			logging.Warnf("valkey: search module 'ver' has unexpected type/value, skipping pre-check")
			return nil
		}
		ver := uint32(verRaw)
		if ver < minPacked {
			return fmt.Errorf("valkey-search %s detected, requires >= %s",
				formatVersion(ver), formatVersion(minPacked))
		}
		return nil
	}
	return fmt.Errorf("valkey-search module not loaded, requires >= %s",
		formatVersion(minPacked))
}

// formatVersion renders a packed version int as "M.N.P" for error messages.
func formatVersion(packed uint32) string {
	return fmt.Sprintf("%d.%d.%d", (packed>>16)&0xFF, (packed>>8)&0xFF, packed&0xFF)
}
