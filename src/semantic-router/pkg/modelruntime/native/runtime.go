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

// Runtime owns one generation's preparation metadata. Pool can be shared with
// the preceding generation; each returned task owns an independent reference.
type Runtime struct {
	Pool      *binding.Pool
	registry  *binding.Registry
	sequence  *binding.Task[string, tasks.LabelDistribution]
	tokens    *binding.Task[string, tasks.TokenClassificationResult]
	grounded  *binding.Task[tasks.GroundedTextRequest, tasks.TokenClassificationResult]
	pair      *binding.Task[tasks.TextPairRequest, tasks.LabelDistribution]
	mu        sync.Mutex
	artifacts map[string]string
}

func New(pool *binding.Pool) *Runtime {
	if pool == nil {
		pool = binding.NewPool()
	}
	registry := binding.NewRegistry(diagnostics.Observe)
	sequence, _ := binding.Register(registry, config.RemoteClassifierContractLabelDistribution, validateText, validateDistribution)
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
	return &Runtime{Pool: pool, registry: registry, sequence: sequence, tokens: tokens, grounded: grounded, pair: pair, artifacts: make(map[string]string)}
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
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	abs, err = filepath.EvalSymlinks(abs)
	if err != nil {
		return "", fmt.Errorf("resolve model artifact: %w", err)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if cached, ok := r.artifacts[abs]; ok {
		return cached, nil
	}
	hash := sha256.New()
	err = filepath.WalkDir(abs, func(path string, entry os.DirEntry, err error) error {
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
		info, err := os.Stat(path)
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("model artifact entry is not a regular file")
		}
		rel, _ := filepath.Rel(abs, path)
		file, err := os.Open(path)
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
	revision := hex.EncodeToString(hash.Sum(nil))
	r.artifacts[abs] = revision
	return revision, nil
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
