package http

import (
	"fmt"
	"io"
)

// ReadLimitedBody reads at most maxBytes from r. If the stream exceeds maxBytes
// it returns an error rather than a silently truncated body, so an oversized or
// malicious upstream response cannot be mis-parsed or exhaust router memory.
//
// maxBytes must be positive; a non-positive ceiling is a caller bug (callers
// resolve their default through a config accessor such as
// config.LooperConfig.GetMaxResponseBytes) and is rejected explicitly so it can
// never silently disable the guard or overflow the maxBytes+1 below.
func ReadLimitedBody(r io.Reader, maxBytes int64) ([]byte, error) {
	if maxBytes <= 0 {
		return nil, fmt.Errorf("read limit must be positive, got %d bytes", maxBytes)
	}
	// Read one byte past the cap so an exactly-at-cap body is accepted while an
	// over-cap body is detectable.
	data, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("response body exceeds limit of %d bytes", maxBytes)
	}
	return data, nil
}

// ReadTruncatedBody reads at most maxBytes from r for diagnostic use, reporting
// whether the stream was longer than the ceiling. Unlike ReadLimitedBody it
// never fails on an oversized stream, because a diagnostic body is never parsed
// and a truncated excerpt is more useful than a dropped one.
func ReadTruncatedBody(r io.Reader, maxBytes int64) (body []byte, truncated bool) {
	if maxBytes <= 0 {
		return nil, false
	}
	data, _ := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if int64(len(data)) > maxBytes {
		return data[:maxBytes], true
	}
	return data, false
}
