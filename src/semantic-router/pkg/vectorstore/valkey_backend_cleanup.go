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
	"context"
	"fmt"
)

// deleteKeysByPrefix uses SCAN to find and DEL all keys matching the given prefix.
// Best-effort: errors during cleanup are silently ignored since the index is already dropped.
func (v *ValkeyBackend) deleteKeysByPrefix(ctx context.Context, prefix string) {
	cursor := "0"
	pattern := prefix + "*"
	for {
		result, err := v.client.CustomCommand(ctx, []string{"SCAN", cursor, "MATCH", pattern, "COUNT", "100"})
		if err != nil {
			return
		}
		arr, ok := result.([]interface{})
		if !ok || len(arr) < 2 {
			return
		}
		cursor = fmt.Sprint(arr[0])

		var keys []string
		if keyList, ok := arr[1].([]interface{}); ok {
			for _, k := range keyList {
				if s, ok := k.(string); ok {
					keys = append(keys, s)
				}
			}
		}
		if len(keys) > 0 {
			_, _ = v.client.Del(ctx, keys)
		}
		if cursor == "0" {
			return
		}
	}
}
