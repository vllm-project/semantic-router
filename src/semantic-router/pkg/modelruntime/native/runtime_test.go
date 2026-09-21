package native

import (
	"context"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type artifactRevisionResult struct {
	revision string
	err      error
}

func TestArtifactRevisionAllowsDistinctPathsConcurrently(t *testing.T) {
	entered := make(chan string, 2)
	release := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	runtime := newRuntimeWithFingerprinter(nil, func(ctx context.Context, path string) (string, error) {
		entered <- path
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-release:
			return path, nil
		}
	})

	results := make(chan artifactRevisionResult, 2)
	for _, path := range []string{"artifact-a", "artifact-b"} {
		go func() {
			revision, err := runtime.artifactRevisionResolved(context.Background(), path)
			results <- artifactRevisionResult{revision: revision, err: err}
		}()
	}
	seen := make(map[string]bool)
	for range 2 {
		select {
		case path := <-entered:
			seen[path] = true
		case <-time.After(time.Second):
			t.Fatal("independent artifact fingerprints did not run concurrently")
		}
	}
	if !seen["artifact-a"] || !seen["artifact-b"] {
		t.Fatalf("fingerprinted paths = %v, want both independent artifacts", seen)
	}
	close(release)
	for range 2 {
		result := <-results
		if result.err != nil || !seen[result.revision] {
			t.Fatalf("artifact revision = %q, error = %v", result.revision, result.err)
		}
	}
}

func TestArtifactRevisionCoalescesCanonicalPathAndCachesSuccess(t *testing.T) {
	artifact := t.TempDir()
	alias := filepath.Join(t.TempDir(), "model")
	if err := os.Symlink(artifact, alias); err != nil {
		t.Fatal(err)
	}
	canonical, err := resolveArtifactPath(artifact)
	if err != nil {
		t.Fatal(err)
	}

	var calls atomic.Int32
	entered := make(chan string, 2)
	release := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	runtime := newRuntimeWithFingerprinter(nil, func(ctx context.Context, path string) (string, error) {
		calls.Add(1)
		entered <- path
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-release:
			return "shared-revision", nil
		}
	})

	results := make(chan artifactRevisionResult, 2)
	for _, path := range []string{artifact, alias} {
		go func() {
			revision, revisionErr := runtime.artifactRevision(context.Background(), path)
			results <- artifactRevisionResult{revision: revision, err: revisionErr}
		}()
	}
	select {
	case path := <-entered:
		if path != canonical {
			t.Fatalf("fingerprint path = %q, want canonical path %q", path, canonical)
		}
	case <-time.After(time.Second):
		t.Fatal("artifact fingerprint did not start")
	}
	waitForArtifactWaiters(t, runtime, canonical, 2)
	if got := calls.Load(); got != 1 {
		t.Fatalf("fingerprint calls = %d, want one coalesced calculation", got)
	}
	close(release)
	for range 2 {
		result := <-results
		if result.err != nil || result.revision != "shared-revision" {
			t.Fatalf("artifact revision = %q, error = %v", result.revision, result.err)
		}
	}
	if revision, revisionErr := runtime.artifactRevision(context.Background(), alias); revisionErr != nil || revision != "shared-revision" {
		t.Fatalf("cached artifact revision = %q, error = %v", revision, revisionErr)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("cached artifact triggered %d fingerprint calls, want one total", got)
	}
}

func TestArtifactRevisionLeaderCancellationKeepsSharedWork(t *testing.T) {
	started := make(chan context.Context, 1)
	release := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	runtime := newRuntimeWithFingerprinter(nil, func(ctx context.Context, _ string) (string, error) {
		started <- ctx
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-release:
			return "shared-revision", nil
		}
	})

	leaderCtx, cancelLeader := context.WithCancel(context.Background())
	leader := make(chan artifactRevisionResult, 1)
	go func() {
		revision, err := runtime.artifactRevisionResolved(leaderCtx, "artifact")
		leader <- artifactRevisionResult{revision: revision, err: err}
	}()
	workCtx := <-started

	waiter := make(chan artifactRevisionResult, 1)
	go func() {
		revision, err := runtime.artifactRevisionResolved(context.Background(), "artifact")
		waiter <- artifactRevisionResult{revision: revision, err: err}
	}()
	waitForArtifactWaiters(t, runtime, "artifact", 2)
	cancelLeader()
	select {
	case result := <-leader:
		if !errors.Is(result.err, context.Canceled) {
			t.Fatalf("canceled leader error = %v", result.err)
		}
	case <-time.After(time.Second):
		t.Fatal("canceled leader did not return promptly")
	}
	select {
	case <-workCtx.Done():
		t.Fatal("one canceled waiter stopped work still needed by another caller")
	default:
	}

	close(release)
	result := <-waiter
	if result.err != nil || result.revision != "shared-revision" {
		t.Fatalf("remaining waiter revision = %q, error = %v", result.revision, result.err)
	}
}

func TestArtifactRevisionWaitsForAbandonedFlightBeforeRetry(t *testing.T) {
	var calls atomic.Int32
	firstStarted := make(chan struct{})
	allowFirstReturn := make(chan struct{})
	secondStarted := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-allowFirstReturn:
		default:
			close(allowFirstReturn)
		}
	})
	runtime := newRuntimeWithFingerprinter(nil, func(ctx context.Context, _ string) (string, error) {
		if calls.Add(1) == 1 {
			close(firstStarted)
			<-ctx.Done()
			<-allowFirstReturn
			return "", ctx.Err()
		}
		close(secondStarted)
		return "retry-revision", nil
	})

	firstCtx, cancelFirst := context.WithCancel(context.Background())
	first := make(chan artifactRevisionResult, 1)
	go func() {
		revision, err := runtime.artifactRevisionResolved(firstCtx, "artifact")
		first <- artifactRevisionResult{revision: revision, err: err}
	}()
	<-firstStarted
	cancelFirst()
	result := <-first
	if !errors.Is(result.err, context.Canceled) {
		t.Fatalf("abandoned waiter error = %v", result.err)
	}
	waitForArtifactWaiters(t, runtime, "artifact", 0)

	retry := make(chan artifactRevisionResult, 1)
	go func() {
		revision, err := runtime.artifactRevisionResolved(context.Background(), "artifact")
		retry <- artifactRevisionResult{revision: revision, err: err}
	}()
	select {
	case <-secondStarted:
		t.Fatal("retry started a second same-path worker before the abandoned flight exited")
	case <-time.After(25 * time.Millisecond):
	}
	close(allowFirstReturn)
	select {
	case <-secondStarted:
	case <-time.After(time.Second):
		t.Fatal("retry did not start after the abandoned flight exited")
	}
	result = <-retry
	if result.err != nil || result.revision != "retry-revision" {
		t.Fatalf("retried artifact revision = %q, error = %v", result.revision, result.err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("fingerprint calls = %d, want canceled attempt and one retry", got)
	}
}

func TestArtifactRevisionRetriesAfterFingerprintFailure(t *testing.T) {
	var calls atomic.Int32
	runtime := newRuntimeWithFingerprinter(nil, func(context.Context, string) (string, error) {
		if calls.Add(1) == 1 {
			return "", errors.New("fingerprint failed")
		}
		return "retry-revision", nil
	})
	if _, err := runtime.artifactRevisionResolved(context.Background(), "artifact"); err == nil {
		t.Fatal("fingerprint failure was accepted")
	}
	revision, err := runtime.artifactRevisionResolved(context.Background(), "artifact")
	if err != nil || revision != "retry-revision" {
		t.Fatalf("retried artifact revision = %q, error = %v", revision, err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("fingerprint calls = %d, want failed attempt and one retry", got)
	}
}

func TestArtifactFingerprintDoesNotBlockTaskRegistration(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	runtime := newRuntimeWithFingerprinter(nil, func(ctx context.Context, _ string) (string, error) {
		close(entered)
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-release:
			return "revision", nil
		}
	})
	artifact := make(chan artifactRevisionResult, 1)
	go func() {
		revision, err := runtime.artifactRevisionResolved(context.Background(), "artifact")
		artifact <- artifactRevisionResult{revision: revision, err: err}
	}()
	<-entered

	registered := make(chan error, 1)
	go func() {
		if _, err := runtime.embeddingTask(); err != nil {
			registered <- err
			return
		}
		_, err := runtime.relevanceTask()
		registered <- err
	}()
	select {
	case err := <-registered:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("artifact fingerprint blocked lazy task registration")
	}
	close(release)
	result := <-artifact
	if result.err != nil || result.revision != "revision" {
		t.Fatalf("artifact revision = %q, error = %v", result.revision, result.err)
	}
}

func waitForArtifactWaiters(t *testing.T, runtime *Runtime, path string, want int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		runtime.artifactMu.Lock()
		flight, ok := runtime.artifactFlights[path]
		got := -1
		if ok {
			got = flight.waiters
		}
		runtime.artifactMu.Unlock()
		if ok && got == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("artifact flight waiters = %d, exists = %t, want %d", got, ok, want)
		}
		time.Sleep(time.Millisecond)
	}
}

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

func TestOperatingPointRejectsArtifactReplacementInsideGeneration(t *testing.T) {
	path := t.TempDir()
	file := filepath.Join(path, "model.safetensors")
	if err := os.WriteFile(file, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}
	runtime := New(nil)
	ctx := context.Background()
	if err := runtime.verifyPreparedArtifact(ctx, path); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte("replacement"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := runtime.verifyPreparedArtifact(ctx, path); err == nil {
		t.Fatal("new policy could reuse old cached owner identity")
	}
	if err := New(runtime.Pool).verifyPreparedArtifact(ctx, path); err != nil {
		t.Fatal("new generation could not prepare replacement", err)
	}
}
