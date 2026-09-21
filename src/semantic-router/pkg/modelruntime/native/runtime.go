// Package native adapts owned inference libraries to typed task bindings.
// Model-family knowledge stays here rather than in routing policy.
package native

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/diagnostics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type artifactFingerprinter func(context.Context, string) (string, error)

type artifactFlight struct {
	done     chan struct{}
	cancel   context.CancelFunc
	waiters  int
	revision string
	err      error
}

// Runtime owns one generation's preparation metadata. Pool can be shared with
// the preceding generation; each returned task owns an independent reference.
type Runtime struct {
	Pool            *binding.Pool
	registry        *binding.Registry
	inventory       *binding.Inventory
	prepared        *binding.PreparedTasks
	operatingPoints map[*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores]]*OperatingPointScorer
	sequence        *binding.Task[string, tasks.LabelDistribution]
	scores          *binding.Task[string, tasks.LabelScores]
	sequenceWindows *binding.Task[tasks.TextWindowsRequest, tasks.WindowedLabelDistribution]
	scoreWindows    *binding.Task[tasks.TextWindowsRequest, tasks.WindowedLabelScores]
	tokens          *binding.Task[string, tasks.TokenClassificationResult]
	tokenWindows    *binding.Task[tasks.TextWindowsRequest, tasks.WindowedTokenClassification]
	grounded        *binding.Task[tasks.GroundedTextRequest, tasks.TokenClassificationResult]
	pair            *binding.Task[tasks.TextPairRequest, tasks.LabelDistribution]
	mu              sync.Mutex
	artifactMu      sync.Mutex
	artifacts       map[string]string
	artifactFlights map[string]*artifactFlight
	fingerprint     artifactFingerprinter
}

func New(pool *binding.Pool) *Runtime {
	return newRuntimeWithFingerprinter(pool, fingerprintArtifact)
}

func newRuntimeWithFingerprinter(pool *binding.Pool, fingerprint artifactFingerprinter) *Runtime {
	if pool == nil {
		pool = binding.NewPool()
	}
	inventory := binding.NewInventory()
	prepared := binding.NewPreparedTasks()
	registry := binding.NewRegistry(func(event binding.Event) {
		inventory.Observe(event)
		prepared.Observe(event)
		diagnostics.Observe(event)
	})
	sequence, _ := binding.Register(registry, config.RemoteClassifierContractLabelDistribution, validateText, validateDistribution)
	scores, _ := binding.Register(registry, config.RemoteClassifierContractLabelScores, validateText, func(_ string, result tasks.LabelScores) error { return tasks.ValidateLabelScores(result.Scores) })
	sequenceWindows, _ := binding.RegisterTask(registry, "windowed_label_distribution.v1", config.RemoteClassifierContractLabelDistribution, validateWindowInput, validateWindowDistribution)
	scoreWindows, _ := binding.RegisterTask(registry, "windowed_label_scores.v1", config.RemoteClassifierContractLabelScores, validateWindowInput, validateWindowScores)
	tokenWindows, _ := binding.RegisterTask(registry, "windowed_token_spans.v1", config.RemoteClassifierContractTokenSpans, validateWindowInput, validateWindowTokens)
	tokens, _ := binding.Register(registry, config.RemoteClassifierContractTokenSpans, validateText, validateSpans)
	grounded, _ := binding.RegisterTask(registry, "grounded_text.v1", config.RemoteClassifierContractTokenSpans, func(input tasks.GroundedTextRequest) error {
		if err := validateText(input.Context); err != nil {
			return err
		}
		return validateText(input.Answer)
	}, func(input tasks.GroundedTextRequest, output tasks.TokenClassificationResult) error {
		return validateSpans(input.Answer, output)
	})
	pair, _ := binding.Register(registry, "text_pair_distribution.v1", func(input tasks.TextPairRequest) error {
		if err := validateText(input.Premise); err != nil {
			return err
		}
		return validateText(input.Hypothesis)
	}, func(_ tasks.TextPairRequest, output tasks.LabelDistribution) error {
		return validateDistribution("", output)
	})
	return &Runtime{
		Pool:            pool,
		registry:        registry,
		inventory:       inventory,
		prepared:        prepared,
		operatingPoints: make(map[*binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores]]*OperatingPointScorer),
		sequence:        sequence,
		scores:          scores,
		sequenceWindows: sequenceWindows,
		scoreWindows:    scoreWindows,
		tokens:          tokens,
		tokenWindows:    tokenWindows,
		grounded:        grounded,
		pair:            pair,
		artifacts:       make(map[string]string),
		artifactFlights: make(map[string]*artifactFlight),
		fingerprint:     fingerprint,
	}
}

// ObserveBinding also admits typed external connectors to this generation's
// inventory without exposing their endpoint or resource compatibility key.
func (r *Runtime) ObserveBinding(event binding.Event) {
	r.inventory.Observe(event)
	r.prepared.Observe(event)
	diagnostics.Observe(event)
}

func (r *Runtime) PreparedBindings() []binding.PreparedBinding {
	return r.inventory.Snapshot()
}

func validateText(text string) error {
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("model input text must not be empty")
	}
	return nil
}

func validateDistribution(_ string, result tasks.LabelDistribution) error {
	if len(result.Probabilities) == 0 {
		return fmt.Errorf("label distribution is empty")
	}
	var sum float64
	for _, value := range result.Probabilities {
		p := float64(value)
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			return fmt.Errorf("label probability is outside [0,1]")
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-3 {
		return fmt.Errorf("label probabilities do not sum to one")
	}
	return nil
}

func validateSpans(text string, result tasks.TokenClassificationResult) error {
	if !utf8.ValidString(text) {
		return fmt.Errorf("token span input must be valid UTF-8")
	}
	limit := len(text)
	if result.TruncatedAt != nil {
		limit = *result.TruncatedAt
		if limit < 0 || limit > len(text) || !utf8.ValidString(text[:limit]) {
			return fmt.Errorf("token span truncation must be an input byte boundary")
		}
	}
	previousEnd := 0
	for _, span := range result.Entities {
		if span.EntityType == "" || span.Start < previousEnd || span.End <= span.Start || span.End > limit || text[span.Start:span.End] != span.Text || !utf8.ValidString(span.Text) {
			return fmt.Errorf("token span does not match its input byte range")
		}
		previousEnd = span.End
		if result.HasScores() && (math.IsNaN(float64(span.Confidence)) || span.Confidence < 0 || span.Confidence > 1) {
			return fmt.Errorf("token span probability is outside [0,1]")
		}
	}
	if result.Summary != nil {
		if result.SummarySemantics == nil || result.SummarySemantics.Unit == "" {
			return fmt.Errorf("token aggregate score semantics are unavailable")
		}
		if err := result.SummarySemantics.Validate(*result.Summary); err != nil {
			return err
		}
	}
	return nil
}

func taskIdentity(spec config.ResolvedModelBinding) binding.Identity {
	return binding.Identity{Recipe: string(spec.Recipe), Name: spec.Name, Deployment: spec.Binding.Deployment, Contract: spec.Binding.Contract, Adapter: spec.Binding.Adapter, Head: spec.Binding.Head}
}

func resourceAdmission(spec config.ResolvedModelBinding) (string, admission.Admissioner) {
	data, _ := json.Marshal(spec.Admission)
	if spec.Admission.MaxConcurrency == 0 {
		return string(data), admission.Noop{}
	}
	return string(data), admission.NewSemaphore(spec.Admission.MaxConcurrency, spec.Admission.MaxQueue, time.Duration(spec.Admission.QueueTimeoutMs)*time.Millisecond, admission.Overflow(spec.Admission.OnOverflow))
}

// artifactRevision fingerprints actual files during preparation, including
// weights, configuration and tokenizer. An in-place replacement at the same
// path cannot reuse the preceding generation's model accidentally.
func (r *Runtime) artifactRevision(ctx context.Context, path string) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	abs, err := resolveArtifactPath(path)
	if err != nil {
		return "", err
	}
	return r.artifactRevisionResolved(ctx, abs)
}

func resolveArtifactPath(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	abs, err = filepath.EvalSymlinks(abs)
	if err != nil {
		return "", fmt.Errorf("resolve model artifact: %w", err)
	}
	return abs, nil
}

// artifactRevisionResolved coalesces one calculation per resolved artifact
// path. An abandoned calculation remains registered until its worker exits, so
// a retry cannot overlap it or consume its canceled result.
func (r *Runtime) artifactRevisionResolved(ctx context.Context, abs string) (string, error) {
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}

		r.artifactMu.Lock()
		if cached, ok := r.artifacts[abs]; ok {
			r.artifactMu.Unlock()
			return cached, nil
		}
		if flight, ok := r.artifactFlights[abs]; ok {
			if flight.waiters == 0 {
				done := flight.done
				r.artifactMu.Unlock()
				select {
				case <-ctx.Done():
					return "", ctx.Err()
				case <-done:
					continue
				}
			}
			flight.waiters++
			r.artifactMu.Unlock()
			return r.waitForArtifactRevision(ctx, abs, flight)
		}

		// The shared calculation has its own lifetime. A caller only cancels it
		// after becoming the final waiter for this artifact.
		workCtx, cancel := context.WithCancel(context.Background())
		flight := &artifactFlight{
			done:    make(chan struct{}),
			cancel:  cancel,
			waiters: 1,
		}
		r.artifactFlights[abs] = flight
		r.artifactMu.Unlock()

		go r.runArtifactFingerprint(workCtx, abs, flight)
		return r.waitForArtifactRevision(ctx, abs, flight)
	}
}

func (r *Runtime) waitForArtifactRevision(ctx context.Context, abs string, flight *artifactFlight) (string, error) {
	select {
	case <-ctx.Done():
		r.detachArtifactWaiter(abs, flight)
		return "", ctx.Err()
	case <-flight.done:
		return flight.revision, flight.err
	}
}

func (r *Runtime) detachArtifactWaiter(abs string, flight *artifactFlight) {
	r.artifactMu.Lock()
	defer r.artifactMu.Unlock()
	if current, ok := r.artifactFlights[abs]; !ok || current != flight || flight.waiters == 0 {
		return
	}
	flight.waiters--
	if flight.waiters == 0 {
		flight.cancel()
	}
}

func (r *Runtime) runArtifactFingerprint(ctx context.Context, abs string, flight *artifactFlight) {
	revision, err := r.fingerprint(ctx, abs)

	r.artifactMu.Lock()
	flight.revision = revision
	flight.err = err
	if current, ok := r.artifactFlights[abs]; ok && current == flight {
		// A calculation abandoned by every waiter must not publish even if its
		// filesystem operation happened to finish after cancellation.
		if err == nil && flight.waiters > 0 {
			r.artifacts[abs] = revision
		}
		delete(r.artifactFlights, abs)
	}
	close(flight.done)
	r.artifactMu.Unlock()
	flight.cancel()
}

// fingerprintArtifact is also used to verify an already prepared generation;
// it uses the same identity framing and file traversal as normal preparation.
func fingerprintArtifact(ctx context.Context, abs string) (string, error) {
	directory, err := os.OpenRoot(filepath.Dir(abs))
	if err != nil {
		return "", fmt.Errorf("open model artifact directory: %w", err)
	}
	defer directory.Close()
	start := filepath.Base(abs)
	info, err := directory.Stat(start)
	if err != nil {
		return "", fmt.Errorf("stat model artifact: %w", err)
	}
	if info.IsDir() {
		// Open the snapshot directory by absolute path so nested symlink targets
		// (for example HF ../../blobs/...) resolve against the cache root.
		artifactDirectory, openErr := os.OpenRoot(filepath.Join(directory.Name(), start))
		if openErr != nil {
			return "", fmt.Errorf("open model artifact: %w", openErr)
		}
		defer artifactDirectory.Close()
		directory, start = artifactDirectory, "."
	}
	hash := sha256.New()
	err = fs.WalkDir(directory.FS(), start, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if contextErr := ctx.Err(); contextErr != nil {
			return contextErr
		}
		if entry.IsDir() {
			return nil
		}
		// ONNX external tensors can use arbitrary filenames. Restricting this
		// to recognized extensions would miss a changed tensor payload.
		rel := filepath.FromSlash(path)
		if start != "." {
			rel = "." // A direct graph/file artifact has no directory prefix.
		}
		file, err := openArtifactFile(directory, filepath.FromSlash(path), entry.Type())
		if err != nil {
			return err
		}
		fileHash := sha256.New()
		_, copyErr := io.Copy(fileHash, contextReader{ctx: ctx, reader: file})
		closeErr := file.Close()
		if copyErr != nil {
			return copyErr
		}
		_, _ = io.WriteString(hash, rel+"\x00"+hex.EncodeToString(fileHash.Sum(nil))+"\x00")
		return closeErr
	})
	if err != nil {
		return "", fmt.Errorf("fingerprint model artifact: %w", err)
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func openArtifactFile(directory *os.Root, name string, mode fs.FileMode) (*os.File, error) {
	if mode&os.ModeSymlink != 0 {
		// HF snapshots intentionally link to ../../blobs. Resolve that explicit
		// target, then anchor its open to its own parent instead of allowing
		// arbitrary path-based opens while walking the artifact directory.
		target, err := directory.Readlink(name)
		if err != nil {
			return nil, err
		}
		if !filepath.IsAbs(target) {
			target = filepath.Join(directory.Name(), filepath.Dir(name), target)
		}
		target, err = filepath.EvalSymlinks(target)
		if err != nil {
			return nil, err
		}
		blobDirectory, err := os.OpenRoot(filepath.Dir(target))
		if err != nil {
			return nil, err
		}
		defer blobDirectory.Close()
		directory, name = blobDirectory, filepath.Base(target)
	}
	info, err := directory.Stat(name)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("model artifact entry is not a regular file")
	}
	file, err := directory.Open(name)
	if err != nil {
		return nil, err
	}
	// Validate the descriptor being hashed as well as the pre-open check,
	// which avoids opening known special files such as FIFOs.
	info, err = file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		_ = file.Close()
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("model artifact entry is not a regular file")
	}
	return file, nil
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(p)
}

// Failed preparation has no effective provider/device facts to advertise.
// Keep the binding identity and error class without logging input or error bodies.
func observePreparationFailure(spec config.ResolvedModelBinding, err error) {
	if err != nil {
		diagnostics.Observe(binding.Event{Identity: taskIdentity(spec), State: "failed", Error: err})
	}
}
