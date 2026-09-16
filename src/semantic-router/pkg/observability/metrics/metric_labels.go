package metrics

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"

// labelOrUnknown keeps an empty string out of a metric label. An empty label
// renders as a blank series that reads like a scrape fault rather than a
// missing value, so it is folded onto the shared unknown label instead.
//
// It lives here rather than beside any one feature's metrics: several files
// call it, and a helper that a whole package depends on should not sit in the
// file for a single signal.
func labelOrUnknown(value string) string {
	if value == "" {
		return consts.UnknownLabel
	}
	return value
}
