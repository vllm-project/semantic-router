package http

import (
	"strings"
	"testing"
)

func TestReadLimitedBody_ExceedsCapReturnsError(t *testing.T) {
	body := strings.NewReader(strings.Repeat("a", 100))

	_, err := ReadLimitedBody(body, 10)

	if err == nil {
		t.Fatal("expected an error when the body exceeds the cap, got nil")
	}
}

func TestReadLimitedBody_WithinCapReturnsFullBody(t *testing.T) {
	body := strings.NewReader("hello world")

	data, err := ReadLimitedBody(body, 1024)
	if err != nil {
		t.Fatalf("unexpected error for a within-cap body: %v", err)
	}
	if string(data) != "hello world" {
		t.Errorf("body = %q, want %q", string(data), "hello world")
	}
}

func TestReadLimitedBody_ExactlyAtCapIsAllowed(t *testing.T) {
	body := strings.NewReader(strings.Repeat("a", 10))

	data, err := ReadLimitedBody(body, 10)
	if err != nil {
		t.Fatalf("unexpected error for a body exactly at the cap: %v", err)
	}
	if len(data) != 10 {
		t.Errorf("read %d bytes, want 10", len(data))
	}
}

func TestReadLimitedBody_DoesNotSilentlyTruncate(t *testing.T) {
	body := strings.NewReader(strings.Repeat("x", 50))

	data, err := ReadLimitedBody(body, 10)

	if err == nil {
		t.Fatalf("expected an error, got a %d-byte body with no error", len(data))
	}
}

func TestReadLimitedBody_OneByteOverCapReturnsError(t *testing.T) {
	body := strings.NewReader(strings.Repeat("a", 11))

	if _, err := ReadLimitedBody(body, 10); err == nil {
		t.Fatal("expected an error for a body of exactly cap+1 bytes, got nil")
	}
}

func TestReadLimitedBody_NonPositiveCapReturnsError(t *testing.T) {
	if _, err := ReadLimitedBody(strings.NewReader(""), 0); err == nil {
		t.Fatal("expected an error for a zero cap, got nil")
	}
	if _, err := ReadLimitedBody(strings.NewReader("x"), -1); err == nil {
		t.Fatal("expected an error for a negative cap, got nil")
	}
}

func TestReadTruncatedBody_WithinCapIsNotTruncated(t *testing.T) {
	body := strings.NewReader("hello")

	data, truncated := ReadTruncatedBody(body, 10)

	if truncated {
		t.Error("truncated = true for a within-cap body, want false")
	}
	if string(data) != "hello" {
		t.Errorf("body = %q, want %q", string(data), "hello")
	}
}

func TestReadTruncatedBody_ExactlyAtCapIsNotTruncated(t *testing.T) {
	body := strings.NewReader(strings.Repeat("a", 10))

	data, truncated := ReadTruncatedBody(body, 10)

	if truncated {
		t.Error("truncated = true for a body exactly at the cap, want false")
	}
	if len(data) != 10 {
		t.Errorf("read %d bytes, want 10", len(data))
	}
}

func TestReadTruncatedBody_OverCapIsCappedNotFailed(t *testing.T) {
	body := strings.NewReader(strings.Repeat("a", 11))

	data, truncated := ReadTruncatedBody(body, 10)

	if !truncated {
		t.Error("truncated = false for a body of cap+1 bytes, want true")
	}
	if len(data) != 10 {
		t.Errorf("read %d bytes, want the body capped at 10", len(data))
	}
}

func TestReadTruncatedBody_NonPositiveCapReadsNothing(t *testing.T) {
	data, truncated := ReadTruncatedBody(strings.NewReader("xxxx"), 0)

	if len(data) != 0 || truncated {
		t.Errorf("got %d bytes truncated=%t, want an empty untruncated body", len(data), truncated)
	}
}
