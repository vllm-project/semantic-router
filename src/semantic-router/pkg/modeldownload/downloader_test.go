package modeldownload

import (
	"errors"
	"testing"
)

func TestIsGatedModelErrorClassifiesOnlyGatedOrAuthFailures(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		err     error
		repoID  string
		hfToken string
		want    bool
	}{
		{
			name:   "nil error is not gated",
			err:    nil,
			repoID: "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
			want:   false,
		},
		{
			name:   "known gated model is skipped even when cli emits only exit status",
			err:    errors.New("exit status 1"),
			repoID: "google/embeddinggemma-300m",
			want:   true,
		},
		{
			name:   "auth failure without token is gated",
			err:    errors.New("401 Unauthorized: authentication required"),
			repoID: "private/model",
			want:   true,
		},
		{
			name:    "auth failure with token is gated",
			err:     errors.New("403 forbidden: gated repo"),
			repoID:  "private/model",
			hfToken: "token",
			want:    true,
		},
		{
			name:   "rate limit without token is not gated",
			err:    errors.New("429 Too Many Requests: rate limit exceeded"),
			repoID: "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
			want:   false,
		},
		{
			name:   "network failure without token is not gated",
			err:    errors.New("dial tcp: lookup huggingface.co: no such host"),
			repoID: "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
			want:   false,
		},
		{
			name:   "plain cli exit without token is not gated for public repo",
			err:    errors.New("exit status 1"),
			repoID: "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
			want:   false,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := IsGatedModelError(tt.err, tt.repoID, tt.hfToken); got != tt.want {
				t.Fatalf("IsGatedModelError() = %v, want %v", got, tt.want)
			}
		})
	}
}
