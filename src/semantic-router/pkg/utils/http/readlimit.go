package http

import (
	"fmt"
	"io"
)

// ReadLimitedBody reads at most maxBytes from r, erroring rather than
// returning a silently truncated body, so an oversized or malicious upstream
// response cannot be mis-parsed or exhaust router memory. A non-positive
// maxBytes is a caller bug and is rejected.
func ReadLimitedBody(r io.Reader, maxBytes int64) ([]byte, error) {
	if maxBytes <= 0 {
		return nil, fmt.Errorf("read limit must be positive, got %d bytes", maxBytes)
	}
	// Read one past the cap so an at-cap body is accepted and an over-cap body
	// is detectable.
	data, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("response body exceeds limit of %d bytes", maxBytes)
	}
	return data, nil
}

// ReadTruncatedBody reads at most maxBytes for diagnostic use, reporting
// whether the stream was longer. Unlike ReadLimitedBody it never fails: a
// diagnostic body is never parsed, so an excerpt beats dropping it.
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
