package modeldownload

import (
	"errors"
	"testing"
)

const exitStatus = "exit status 1"

type gatedModelErrorCase struct {
	name      string
	err       error
	cliOutput string
	repoID    string
	want      bool
}

// gatedModelErrorCases is declared at package scope (not inside the test) so the
// table can grow without tripping the funlen limit on the test function.
var gatedModelErrorCases = []gatedModelErrorCase{
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
	{
		// The #2107 GKE scenario: a public download that made progress (tqdm
		// prints "403M/896M") and then got rate-limited. The bare "403" in the
		// byte count must NOT be read as a gated 403.
		name:      "public repo progress bytes then rate limit is not gated",
		err:       errors.New(exitStatus),
		cliOutput: "Downloading model.safetensors: 45%|xxxx| 403M/896M [00:12<00:14, 34.2MB/s]\n429 Client Error: Too Many Requests for url: https://huggingface.co/api/models/llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
		repoID:    "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
		want:      false,
	},
	{
		// A gemma-derived PUBLIC mirror hitting a transient 429. The repoID
		// allowlist must stay a last resort (skipped because stderr is present),
		// otherwise "gemma" in the name soft-skips a public transient failure.
		name:      "gemma-named public mirror rate limit is not gated",
		err:       errors.New(exitStatus),
		cliOutput: "429 Client Error: Too Many Requests for url",
		repoID:    "some-org/gemma-derived-public-mirror",
		want:      false,
	},
	{
		name:      "multi-line GatedRepoError stderr is gated",
		err:       errors.New(exitStatus),
		cliOutput: "Traceback (most recent call last):\n  File \"hf.py\", line 1, in <module>\nhuggingface_hub.errors.GatedRepoError: 403 Client Error. (Request ID: Root=1-abc)\nCannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/config.json.\nAccess to model meta-llama/Llama-3.1-8B is restricted. You must be authenticated to access it.",
		repoID:    "meta-llama/Llama-3.1-8B",
		want:      true,
	},
	{
		// The exact string the Rust `hf` CLI emits for a gated repo. stderr is
		// present, so this must be caught by the "requires approval" anchor, NOT
		// by the gemma repoID allowlist (which is now a last resort). Regression
		// guard for the CI embeddinggemma download.
		name:      "hf CLI access-denied on gated repo is gated",
		err:       errors.New(exitStatus),
		cliOutput: "Fetching 19 files:   0%|          | 0/19 [00:00<?, ?it/s]Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nError: Access denied. This repository requires approval.\nHint: set HF_DEBUG=1 as environment variable for full traceback.",
		repoID:    "google/embeddinggemma-300m",
		want:      true,
	},
}

func TestIsGatedModelErrorClassifiesOnlyGatedOrAuthFailures(t *testing.T) {
	t.Parallel()

	for _, tt := range gatedModelErrorCases {
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
