package native

import (
	"context"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestArtifactFingerprintIncludesExternalTensorsAcrossGenerations(t *testing.T) {
	artifact := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(artifact, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("model.onnx", "graph with external tensors")
	write("arbitrary_tensor_name", "first weights")
	first := New(nil)
	before, err := first.artifactRevision(context.Background(), artifact)
	if err != nil {
		t.Fatal(err)
	}
	write("arbitrary_tensor_name", "second weights")
	after, err := New(first.Pool).artifactRevision(context.Background(), artifact)
	if err != nil || before == after {
		t.Fatalf("same-path tensor replacement reused identity: %q, %v", after, err)
	}
	alias := filepath.Join(t.TempDir(), "model")
	if linkErr := os.Symlink(artifact, alias); linkErr != nil {
		t.Fatal(linkErr)
	}
	aliased, err := New(nil).artifactRevision(context.Background(), alias)
	if err != nil || aliased != after {
		t.Fatalf("artifact root symlink lost file content: %q, %v", aliased, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := New(nil).artifactRevision(ctx, artifact); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled preparation hashed model files: %v", err)
	}
}

func TestArtifactFingerprintFramesEachFile(t *testing.T) {
	first, second := t.TempDir(), t.TempDir()
	for name, content := range map[string]string{"a": "payload", "b": "tail"} {
		if err := os.WriteFile(filepath.Join(first, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(second, "a"), []byte("payloadb\x00tail"), 0o600); err != nil {
		t.Fatal(err)
	}
	runtime := New(nil)
	one, err := runtime.artifactRevision(context.Background(), first)
	if err != nil {
		t.Fatal(err)
	}
	two, err := runtime.artifactRevision(context.Background(), second)
	if err != nil || one == two {
		t.Fatalf("different artifact trees shared an ambiguous fingerprint: %v", err)
	}
}

func TestArtifactFingerprintReadsExternalSnapshotBlobs(t *testing.T) {
	cache := t.TempDir()
	snapshot := filepath.Join(cache, "snapshots", "revision")
	blobs := filepath.Join(cache, "blobs")
	for _, directory := range []string{snapshot, blobs} {
		if err := os.MkdirAll(directory, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	blob := filepath.Join(blobs, "weights")
	link := filepath.Join(snapshot, "model.safetensors")
	if err := os.Symlink(filepath.Join("..", "..", "blobs", "weights"), link); err != nil {
		t.Fatal(err)
	}
	plain := t.TempDir()
	var previous string
	for _, content := range []string{"first weights", "replacement weights"} {
		for _, path := range []string{blob, filepath.Join(plain, "model.safetensors")} {
			if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		linked, err := New(nil).artifactRevision(context.Background(), snapshot)
		if err != nil {
			t.Fatal(err)
		}
		copied, err := New(nil).artifactRevision(context.Background(), plain)
		if err != nil || linked != copied || linked == previous {
			t.Fatalf("external blob content was not fingerprinted: linked=%q copied=%q previous=%q err=%v", linked, copied, previous, err)
		}
		previous = linked
	}
}

func TestArtifactFingerprintDirectFileAndFailedPreparation(t *testing.T) {
	artifact := t.TempDir()
	graph := filepath.Join(artifact, "model.onnx")
	if err := os.WriteFile(graph, []byte("graph"), 0o600); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(t.TempDir(), "graph.onnx")
	if err := os.Symlink(graph, alias); err != nil {
		t.Fatal(err)
	}
	fileRevision, err := New(nil).artifactRevision(context.Background(), graph)
	if err != nil {
		t.Fatal(err)
	}
	aliasRevision, err := New(nil).artifactRevision(context.Background(), alias)
	if err != nil || aliasRevision != fileRevision {
		t.Fatalf("direct file alias changed fingerprint: %q, %v", aliasRevision, err)
	}
	for _, target := range []string{t.TempDir(), filepath.Join(t.TempDir(), "missing")} {
		link := filepath.Join(artifact, "invalid")
		if err := os.Symlink(target, link); err != nil {
			t.Fatal(err)
		}
		runtime := New(nil)
		if _, hashErr := runtime.artifactRevision(context.Background(), artifact); hashErr == nil || len(runtime.artifacts) != 0 {
			t.Fatalf("invalid artifact was accepted or cached: %v", hashErr)
		}
		if err := os.Remove(link); err != nil {
			t.Fatal(err)
		}
		if _, hashErr := runtime.artifactRevision(context.Background(), artifact); hashErr != nil {
			t.Fatalf("failed preparation prevented retry: %v", hashErr)
		}
	}
	runtime := New(nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, hashErr := runtime.artifactRevision(ctx, artifact); !errors.Is(hashErr, context.Canceled) || len(runtime.artifacts) != 0 {
		t.Fatalf("canceled preparation was accepted or cached: %v", hashErr)
	}
	if _, hashErr := runtime.artifactRevision(context.Background(), artifact); hashErr != nil {
		t.Fatalf("canceled preparation prevented retry: %v", hashErr)
	}
}

func TestFingerprintReaderStopsAfterCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	reader := contextReader{ctx: ctx, reader: readerFunc(func(p []byte) (int, error) {
		cancel()
		return copy(p, "one read"), nil
	})}
	if _, err := io.Copy(io.Discard, reader); !errors.Is(err, context.Canceled) {
		t.Fatalf("file hashing ignored cancellation: %v", err)
	}
}

type readerFunc func([]byte) (int, error)

func (r readerFunc) Read(p []byte) (int, error) { return r(p) }

func TestNativeSpansValidateUnicodePartialAndScoreSemantics(t *testing.T) {
	text := "前🙂 claim tail"
	available := true
	cut := len("前🙂 claim")
	valid := tasks.TokenClassificationResult{Entities: []tasks.TokenEntity{{EntityType: "unsupported", Start: len("前🙂 "), End: cut, Text: "claim", Confidence: 0.7}}, ScoresAvailable: &available, TruncatedAt: &cut}
	if err := validateSpans(text, valid); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"mid-rune", "overlap", "past-truncation", "invalid-score", "unknown-summary"} {
		t.Run(name, func(t *testing.T) {
			result := valid
			result.Entities = append([]tasks.TokenEntity(nil), valid.Entities...)
			switch name {
			case "mid-rune":
				result.Entities[0].Start, result.Entities[0].End = 1, 3
				result.Entities[0].Text = text[1:3]
			case "overlap":
				result.Entities = append(result.Entities, result.Entities[0])
			case "past-truncation":
				shortCut := len("前🙂")
				result.TruncatedAt = &shortCut
			case "invalid-score":
				result.Entities[0].Confidence = float32(math.NaN())
			case "unknown-summary":
				result.Summary = &tasks.ScoreResult{Value: 1}
			}
			if err := validateSpans(text, result); err == nil {
				t.Fatal("invalid native spans were accepted")
			}
		})
	}
}
