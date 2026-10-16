package systemone

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// publicOptionNames lists the request options of the public System One API.
const publicOptionNames = "deadline_ms, max_tokens and return_meta"

// removedOptions are former request options and why a request cannot set them.
var removedOptions = map[string]string{
	"profile": "options.profile was removed: batching and numerics are the deployment's engine arguments " +
		"(max_num_seqs, max_num_batched_tokens, batch_window_ms, enable_shared_context, dtype)",
}

// publicOptions checks the value of a request's options member against the
// public System One API and encodes it as the runtime contract's options, so
// a model receives only options that API defines. null is no options. The
// second result is why the options are invalid, or "".
func publicOptions(raw []byte) (json.RawMessage, string) {
	var members map[string]json.RawMessage
	if json.Unmarshal(raw, &members) != nil {
		return nil, "options must be an object"
	}
	var options api.RequestOptions
	for _, name := range slices.Sorted(maps.Keys(members)) {
		value := members[name]
		switch name {
		case "deadline_ms":
			var deadline float64
			if json.Unmarshal(value, &deadline) != nil || !(deadline > 0) || math.IsInf(deadline, 0) {
				return nil, "options.deadline_ms must be a number greater than 0"
			}
			options.DeadlineMs = &deadline
		case "max_tokens":
			var tokens int
			if json.Unmarshal(value, &tokens) != nil || tokens < 1 {
				return nil, "options.max_tokens must be an integer of at least 1"
			}
			options.MaxTokens = &tokens
		case "return_meta":
			var enabled *bool
			if json.Unmarshal(value, &enabled) != nil || enabled == nil {
				return nil, "options.return_meta must be true or false"
			}
			options.ReturnMeta = enabled
		default:
			if reason, removed := removedOptions[name]; removed {
				return nil, reason
			}
			return nil, fmt.Sprintf("options.%s is not a System One option; the options are %s", name, publicOptionNames)
		}
	}
	encoded, err := json.Marshal(options)
	if err != nil {
		return nil, "options must be an object"
	}
	return encoded, ""
}
