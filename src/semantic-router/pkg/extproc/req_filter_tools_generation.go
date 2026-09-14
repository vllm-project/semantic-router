package extproc

import (
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// toolDefinitionsEqual reports whether two ordered tool definitions are
// semantically identical. A retained tool prefix should keep its original
// envelope generation when selection produces the same definitions; any
// order or provider-visible field change must still invalidate replay.
func toolDefinitionsEqual(left, right []llmprotocol.Tool) bool {
	return reflect.DeepEqual(left, right)
}
