package modeldownload

import (
	"errors"
	"testing"
)

func TestIsGatedModelErrorClassifiesOnlyGatedOrAuthFailures(t *testing.T) {
	t.Parallel()

	const exitStatus = "exit status 1"

	tests := []struct {
		name      string
		err       error
		cliOutput string
		repoID    string
		want      bool
	}{
		{
			name:   "nil error is not gated",
			err:    nil,
			repoID: "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
			want:   false,
		},
		{
			name:   "known gated model is skipped when cli emits only exit status",
			err:    errors.New(exitStatus),
			repoID: "google/embeddinggemma-300m",
			want:   true,
		},
		{
			name:      "captured gated access message is skipped",
			err:       errors.New(exitStatus),
			cliOutput: "Access to model private/model is restricted. You must be authenticated to access it.",
			repoID:    "private/model",
			want:      true,
		},
		{
			name:      "captured 401 is gated",
			err:       errors.New(exitStatus),
			cliOutput: "401 Client Error: Unauthorized for url: https://huggingface.co/api/models/private/model",
			repoID:    "private/model",
			want:      true,
		},
		{
			name:      "captured 403 gated repo is gated",
			err:       errors.New(exitStatus),
			cliOutput: "403 Forbidden: you are trying to access a gated repo",
			repoID:    "private/model",
			want:      true,
		},
		{
			name:      "rate limit on public repo is not gated",
			err:       errors.New(exitStatus),
			cliOutput: "429 Client Error: Too Many Requests for url",
			repoID:    "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
			want:      false,
		},
		{
			name:      "invalid identifier 429 on public repo is not gated",
			err:       errors.New(exitStatus),
			cliOutput: "429 Client Error: Too Many Requests for url: https://huggingface.co/api/models/typo/does-not-exist",
			repoID:    "typo/does-not-exist",
			want:      false,
		},
		{
			name:      "missing repo 404 is not gated",
			err:       errors.New(exitStatus),
			cliOutput: "404 Client Error: Repository Not Found for url",
			repoID:    "typo/does-not-exist",
			want:      false,
		},
		{
			name:      "network failure on public repo is not gated",
			err:       errors.New(exitStatus),
			cliOutput: "dial tcp: lookup huggingface.co: no such host",
			repoID:    "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
			want:      false,
		},
		{
			name:   "plain cli exit with no captured output is not gated for public repo",
			err:    errors.New(exitStatus),
			repoID: "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
			want:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := IsGatedModelError(tt.err, tt.cliOutput, tt.repoID); got != tt.want {
				t.Fatalf("IsGatedModelError() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTailWriterRetainsTail(t *testing.T) {
	t.Parallel()

	var w tailWriter
	// Write more than maxCaptureBytes; only the tail must be retained.
	chunk := make([]byte, maxCaptureBytes)
	for i := range chunk {
		chunk[i] = 'a'
	}
	if _, err := w.Write(chunk); err != nil {
		t.Fatalf("Write returned error: %v", err)
	}
	if _, err := w.Write([]byte("GATED_TAIL")); err != nil {
		t.Fatalf("Write returned error: %v", err)
	}

	got := w.String()
	if len(got) > maxCaptureBytes {
		t.Fatalf("tailWriter exceeded cap: len=%d cap=%d", len(got), maxCaptureBytes)
	}
	if got[len(got)-10:] != "GATED_TAIL" {
		t.Fatalf("tailWriter dropped the tail: %q", got[len(got)-10:])
	}
}
