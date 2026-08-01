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

package looper

import (
	"strings"
	"testing"
)

// A single upstream model response must not be read without a byte ceiling:
// the looper endpoint is often itself another router, and an unbounded
// io.ReadAll lets a huge/malicious body OOM the process (amplified N-fold by
// the parallel fan-out algorithms). readLimitedBody enforces that ceiling.

func TestReadLimitedBody_ExceedsCapReturnsError(t *testing.T) {
	body := strings.NewReader(strings.Repeat("a", 100))

	_, err := readLimitedBody(body, 10)

	if err == nil {
		t.Fatal("expected an error when the body exceeds the cap, got nil")
	}
}

func TestReadLimitedBody_WithinCapReturnsFullBody(t *testing.T) {
	body := strings.NewReader("hello world")

	data, err := readLimitedBody(body, 1024)
	if err != nil {
		t.Fatalf("unexpected error for a within-cap body: %v", err)
	}
	if string(data) != "hello world" {
		t.Errorf("body = %q, want %q", string(data), "hello world")
	}
}

func TestReadLimitedBody_ExactlyAtCapIsAllowed(t *testing.T) {
	body := strings.NewReader(strings.Repeat("a", 10))

	data, err := readLimitedBody(body, 10)
	if err != nil {
		t.Fatalf("unexpected error for a body exactly at the cap: %v", err)
	}
	if len(data) != 10 {
		t.Errorf("read %d bytes, want 10", len(data))
	}
}

func TestReadLimitedBody_DoesNotSilentlyTruncate(t *testing.T) {
	// An over-cap body must surface as an explicit error, never a silently
	// truncated (and thus mis-parsed) partial body.
	body := strings.NewReader(strings.Repeat("x", 50))

	data, err := readLimitedBody(body, 10)

	if err == nil {
		t.Fatalf("expected an error, got a %d-byte body with no error", len(data))
	}
}

func TestReadLimitedBody_OneByteOverCapReturnsError(t *testing.T) {
	// The exact boundary the maxBytes+1 LimitReader trick hinges on: a body of
	// exactly cap+1 bytes must be rejected.
	body := strings.NewReader(strings.Repeat("a", 11))

	if _, err := readLimitedBody(body, 10); err == nil {
		t.Fatal("expected an error for a body of exactly cap+1 bytes, got nil")
	}
}

func TestReadLimitedBody_NonPositiveCapReturnsError(t *testing.T) {
	// The helper enforces a positive ceiling; a non-positive cap is a caller
	// bug and must fail explicitly rather than mis-report an empty body.
	if _, err := readLimitedBody(strings.NewReader(""), 0); err == nil {
		t.Fatal("expected an error for a zero cap, got nil")
	}
	if _, err := readLimitedBody(strings.NewReader("x"), -1); err == nil {
		t.Fatal("expected an error for a negative cap, got nil")
	}
}
