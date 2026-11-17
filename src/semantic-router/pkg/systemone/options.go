package systemone

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// publicOptionNames lists the request options of the public System One API.
const publicOptionNames = "deadline_ms, long_context, max_tokens and return_meta"

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
		case "long_context":
			var enabled *bool
			if json.Unmarshal(value, &enabled) != nil || enabled == nil {
				return nil, "options.long_context must be true or false"
			}
			options.LongContext = enabled
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

// requestWait is how long the public API waits for a model it calls directly:
// publicWait, or the request's deadline_ms when longer, at most limit. A
// long-context request without deadline_ms waits limit: reading up to a
// model's max_input_tokens takes minutes, far past publicWait.
func requestWait(options json.RawMessage, limit time.Duration) time.Duration {
	wait := publicWait
	var parsed api.RequestOptions
	if json.Unmarshal(options, &parsed) == nil {
		switch {
		case parsed.DeadlineMs != nil:
			ms := min(*parsed.DeadlineMs, float64(limit/time.Millisecond))
			wait = max(time.Duration(ms*float64(time.Millisecond)), publicWait)
		case parsed.LongContext != nil && *parsed.LongContext:
			wait = limit
		}
	}
	return min(wait, limit)
}

// deadlineExceeded is the message of a request the public API stopped waiting
// for: how long it waited, and how a request waits longer when it can.
func deadlineExceeded(wait, limit time.Duration) string {
	if wait < limit {
		return fmt.Sprintf("no answer within %s; options.deadline_ms waits longer, up to this listener's systemone.max_deadline_ms (%d ms)", wait, limit.Milliseconds())
	}
	return fmt.Sprintf("no answer within %s, this listener's systemone.max_deadline_ms", wait)
}
