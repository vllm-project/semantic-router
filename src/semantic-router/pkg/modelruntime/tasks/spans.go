package tasks

import "errors"

// ErrTokenSpansTruncated marks a valid but incomplete token-spans result. A
// consumer can still use the returned spans, but must apply its partial-input
// policy to the unscanned remainder instead of treating it as a clean scan.
var ErrTokenSpansTruncated = errors.New("token_spans.v1 response is partial: provider truncated its input")
