package configsnapshot

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// fakeRuntime records what the lifecycle asks of the Router.
type fakeRuntime struct {
	validate func(*Candidate) error
	warm     func(context.Context, *Candidate) error
	activate func(*Candidate) error

	warming    atomic.Int32
	maxWarming atomic.Int32
	validated  atomic.Int32
	discarded  atomic.Int32

	mu        sync.Mutex
	activated []uint64
}

func (r *fakeRuntime) Validate(_ context.Context, c *Candidate) error {
	r.validated.Add(1)
	if r.validate != nil {
		return r.validate(c)
	}
	return nil
}

func (r *fakeRuntime) Warm(ctx context.Context, c *Candidate) (Warmed, error) {
	now := r.warming.Add(1)
	defer r.warming.Add(-1)
	for {
		peak := r.maxWarming.Load()
		if now <= peak || r.maxWarming.CompareAndSwap(peak, now) {
			break
		}
	}
	c.Step("model_prepare")
	if r.warm != nil {
		if err := r.warm(ctx, c); err != nil {
			return nil, err
		}
	}
	return &fakeWarmed{runtime: r, candidate: c}, nil
}

type fakeWarmed struct {
	runtime   *fakeRuntime
	candidate *Candidate
}

func (w *fakeWarmed) Activate(context.Context) error {
	if w.runtime.activate != nil {
		if err := w.runtime.activate(w.candidate); err != nil {
			return err
		}
	}
	w.runtime.mu.Lock()
	w.runtime.activated = append(w.runtime.activated, w.candidate.Snapshot().Version())
	w.runtime.mu.Unlock()
	return nil
}

func (w *fakeWarmed) Discard() { w.runtime.discarded.Add(1) }

type recordingReporter struct {
	mu       sync.Mutex
	attempts []Attempt
}

func (r *recordingReporter) ReportConfigAttempt(attempt Attempt, _ error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.attempts = append(r.attempts, attempt)
}

func (r *recordingReporter) last() Attempt {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.attempts[len(r.attempts)-1]
}

func testConfig(hash string) *config.RouterConfig {
	return &config.RouterConfig{DocumentHash: hash}
}

func installed(t *testing.T, runtime Runtime, reporter Reporter) *Manager {
	t.Helper()
	m := NewManager(Options{Runtime: runtime, Reporter: reporter})
	snapshot, err := m.Install(context.Background(), Update{Origin: Origin{Source: SourceStartup}, Config: testConfig("h1")})
	if err != nil || snapshot.Version() != 1 {
		t.Fatalf("Install() = %v, %v", snapshot, err)
	}
	return m
}

func TestApplyActivatesTheNextVersion(t *testing.T) {
	reporter := &recordingReporter{}
	runtime := &fakeRuntime{}
	m := installed(t, runtime, reporter)

	snapshot, err := m.Apply(context.Background(), Update{
		Origin: Origin{Source: SourceFile}, Config: testConfig("h2"), Document: []byte("doc"),
	})
	if err != nil {
		t.Fatalf("Apply() error = %v", err)
	}
	if snapshot.Version() != 2 || snapshot.Hash() != "h2" || string(snapshot.Document()) != "doc" ||
		snapshot.Origin().Source != SourceFile || m.Active() != snapshot {
		t.Fatalf("activated snapshot = v%d %q %+v", snapshot.Version(), snapshot.Hash(), snapshot.Origin())
	}
	status := m.Status()
	if status.Active.Version != 2 || status.Latest.Status != AttemptActive || status.Latest.Version != 2 ||
		status.LastNACK != nil {
		t.Fatalf("status = %+v", status)
	}

	var stages []Stage
	for _, attempt := range reporter.attempts {
		if attempt.ID == status.Latest.ID && (len(stages) == 0 || stages[len(stages)-1] != attempt.Stage) {
			stages = append(stages, attempt.Stage)
		}
	}
	want := []Stage{StageCompile, StageValidate, StageWarm, StageActivate}
	if len(stages) != len(want) {
		t.Fatalf("reported stages = %v, want %v", stages, want)
	}
	for i := range want {
		if stages[i] != want[i] {
			t.Fatalf("reported stages = %v, want %v", stages, want)
		}
	}
	if last := reporter.last(); last.Status != AttemptActive || last.Step != "model_prepare" || last.FinishedAt == nil {
		t.Fatalf("last report = %+v", last)
	}
}

func TestRejectedUpdateKeepsTheActiveSnapshotAndItsVersion(t *testing.T) {
	cause := errors.New("routing_preview.max_concurrency changed")
	runtime := &fakeRuntime{validate: func(*Candidate) error {
		return Reject(StageValidate, CodeRestartRequired, cause)
	}}
	m := installed(t, runtime, nil)
	before := m.Active()

	_, err := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceAPI}, Config: testConfig("h2")})
	if !errors.Is(err, cause) || err.Error() != cause.Error() {
		t.Fatalf("Apply() error = %v, want the cause unchanged", err)
	}
	reasons := ReasonsOf(err)
	if len(reasons) != 1 || reasons[0].Stage != StageValidate || reasons[0].Code != CodeRestartRequired {
		t.Fatalf("reasons = %+v", reasons)
	}
	if m.Active() != before || runtime.discarded.Load() != 0 {
		t.Fatal("a rejected update replaced or discarded the active snapshot")
	}
	status := m.Status()
	if status.LastNACK == nil || status.LastNACK.Stage != StageValidate || status.LastNACK.Hash != "h2" ||
		status.Active.Version != 1 {
		t.Fatalf("status = %+v", status)
	}
	if got := testutil.ToFloat64(metrics.ConfigLastRejection); got <= 0 {
		t.Fatalf("last rejection metric = %v", got)
	}

	runtime.validate = nil
	snapshot, err := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceAPI}, Config: testConfig("h3")})
	if err != nil || snapshot.Version() != 2 {
		t.Fatalf("the next ACK = v%d, %v; want v2: rejections take no version", snapshot.Version(), err)
	}
	if m.Status().LastNACK.Hash != "h2" {
		t.Fatal("an ACK erased the last NACK")
	}
}

func TestRuntimeErrorsAreClassifiedByStage(t *testing.T) {
	for _, tc := range []struct {
		name    string
		runtime *fakeRuntime
		stage   Stage
		code    Code
	}{
		{"validate", &fakeRuntime{validate: func(*Candidate) error { return errors.New("boom") }}, StageValidate, CodeInvalidResource},
		{"warm", &fakeRuntime{warm: func(context.Context, *Candidate) error { return errors.New("boom") }}, StageWarm, CodeBuildFailed},
		{"activate", &fakeRuntime{activate: func(*Candidate) error { return errors.New("boom") }}, StageActivate, CodeActivationFailed},
		{"canceled", &fakeRuntime{warm: func(context.Context, *Candidate) error { return context.Canceled }}, StageWarm, CodeCanceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := installed(t, tc.runtime, nil)
			_, err := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceFile}, Config: testConfig("h2")})
			reasons := ReasonsOf(err)
			if len(reasons) != 1 || reasons[0].Stage != tc.stage || reasons[0].Code != tc.code {
				t.Fatalf("reasons = %+v (err %v)", reasons, err)
			}
			if m.Active().Version() != 1 || tc.runtime.discarded.Load() != 0 {
				t.Fatal("the active snapshot changed")
			}
		})
	}
}

func TestSupersededUpdateIsNeitherACKNorNACK(t *testing.T) {
	m := installed(t, &fakeRuntime{validate: func(c *Candidate) error { return c.Current() }}, nil)
	_, err := m.Apply(context.Background(), Update{
		Origin: Origin{Source: SourceFile}, Config: testConfig("h2"),
		Current: func() error { return ErrSuperseded },
	})
	if !errors.Is(err, ErrSuperseded) || ReasonsOf(err) != nil {
		t.Fatalf("Apply() error = %v", err)
	}
	status := m.Status()
	if status.Latest.Status != AttemptSuperseded || status.LastNACK != nil || status.Active.Version != 1 {
		t.Fatalf("status = %+v", status)
	}
}

func TestCompileRejectionNeverReachesTheRuntime(t *testing.T) {
	runtime := &fakeRuntime{}
	m := installed(t, runtime, nil)
	cfg := testConfig("h2")
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"x"}, Recipe: "missing"}}
	_, err := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceKubernetes}, Config: cfg})
	if reasons := ReasonsOf(err); len(reasons) != 1 || reasons[0].Code != CodeUnresolvedReference {
		t.Fatalf("Apply() error = %v", err)
	}
	if runtime.validated.Load() != 0 {
		t.Fatal("the runtime validated a candidate that does not compile")
	}
	if m.Status().LastNACK.Stage != StageCompile {
		t.Fatalf("last NACK = %+v", m.Status().LastNACK)
	}
}

func TestRejectRecordsASourceFailureOnce(t *testing.T) {
	reporter := &recordingReporter{}
	runtime := &fakeRuntime{warm: func(context.Context, *Candidate) error { return errors.New("model missing") }}
	m := installed(t, runtime, reporter)

	parseErr := errors.New("yaml: line 3: mapping values are not allowed")
	err := m.Reject(Update{Origin: Origin{Source: SourceFile}, Document: []byte("bad: [")}, parseErr)
	if err.Error() != parseErr.Error() {
		t.Fatalf("Reject() error = %q", err)
	}
	nack := m.Status().LastNACK
	if nack == nil || nack.Stage != StageParse || nack.Step != "parse" || nack.Hash != documentHash(nil, []byte("bad: [")) {
		t.Fatalf("last NACK = %+v", nack)
	}

	_, applyErr := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceKubernetes}, Config: testConfig("h2")})
	attempts := m.Status().Latest.ID
	if got := m.Reject(Update{Origin: Origin{Source: SourceKubernetes}}, applyErr); !errors.Is(got, applyErr) {
		t.Fatalf("Reject() of a lifecycle error = %v", got)
	}
	if m.Status().Latest.ID != attempts {
		t.Fatal("a rejection the lifecycle recorded was recorded again")
	}
	if m.Reject(Update{}, nil) != nil {
		t.Fatal("Reject(nil) reported an error")
	}
}

func TestUpdatesRunOneAtATime(t *testing.T) {
	runtime := &fakeRuntime{warm: func(context.Context, *Candidate) error {
		time.Sleep(time.Millisecond)
		return nil
	}}
	m := installed(t, runtime, &recordingReporter{})
	const updates = 8
	var wg sync.WaitGroup
	for i := range updates {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if _, err := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceAPI}, Config: testConfig("h")}); err != nil {
				t.Errorf("Apply(%d) error = %v", i, err)
			}
		}(i)
		go func() { _ = m.Status() }()
	}
	wg.Wait()
	if peak := runtime.maxWarming.Load(); peak != 1 {
		t.Fatalf("%d updates warmed at once", peak)
	}
	seen := make(map[uint64]bool)
	for _, version := range runtime.activated {
		if seen[version] {
			t.Fatalf("version %d activated twice: %v", version, runtime.activated)
		}
		seen[version] = true
	}
	if len(seen) != updates || m.Active().Version() != updates+1 {
		t.Fatalf("activated %v, active v%d", runtime.activated, m.Active().Version())
	}
}

func TestStatusDoesNotWaitForAnUpdate(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	m := installed(t, &fakeRuntime{warm: func(context.Context, *Candidate) error {
		close(entered)
		<-release
		return nil
	}}, nil)
	done := make(chan error, 1)
	go func() {
		_, err := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceFile}, Config: testConfig("h2")})
		done <- err
	}()
	<-entered
	read := make(chan Status, 1)
	go func() { read <- m.Status() }()
	select {
	case status := <-read:
		if status.Latest.Stage != StageWarm || status.Latest.Status != AttemptPreparing || status.Active.Version != 1 {
			t.Fatalf("status while warming = %+v", status)
		}
	case <-time.After(time.Second):
		t.Fatal("Status() waited for the update in progress")
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestCanceledUpdateIsRejectedBeforeCompile(t *testing.T) {
	runtime := &fakeRuntime{}
	m := installed(t, runtime, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := m.Apply(ctx, Update{Origin: Origin{Source: SourceKubernetes}, Config: testConfig("h2")})
	if !errors.Is(err, context.Canceled) || ReasonsOf(err)[0].Code != CodeCanceled || runtime.validated.Load() != 0 {
		t.Fatalf("Apply() error = %v", err)
	}
}

func TestApplyWithoutRuntimeIsRejected(t *testing.T) {
	m := NewManager(Options{})
	_, err := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceFile}, Config: testConfig("h1")})
	if reasons := ReasonsOf(err); len(reasons) != 1 || reasons[0].Stage != StageValidate {
		t.Fatalf("Apply() error = %v", err)
	}
	if m.Active() != nil {
		t.Fatal("a manager without a runtime activated a snapshot")
	}
}
