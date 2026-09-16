package extproc

import (
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// toolDefinitionsEqual reports whether two ordered tool definitions are
// identical from the provider's perspective. A retained tool prefix should
// keep its original envelope generation only when selection produces the same
// serialized definitions; any order or provider-visible field change must
// still invalidate replay.
func toolDefinitionsEqual(left, right []llmprotocol.Tool) bool {
	return reflect.DeepEqual(left, right)
}
