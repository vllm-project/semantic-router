package configsnapshot

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// ErrSuperseded reports an update that a newer document of the same source
// replaced before it could activate. It is neither an ACK nor a NACK: the
// newer document is applied on its own.
var ErrSuperseded = errors.New("config reload superseded by a newer source document")

// Update is a candidate configuration offered to the lifecycle.
type Update struct {
	Origin Origin
	// Config is the parsed candidate.
	Config *config.RouterConfig
	// Document is the exact document Config was parsed from; nil when the
	// source has none.
	Document []byte
	// Current, when set, reports whether the update is still the source's
	// latest document; it returns ErrSuperseded once a newer one replaced it.
	// The runtime checks it before preparing models and again at activation.
	Current func() error
}

// Runtime prepares and activates snapshots on the serving Router.
type Runtime interface {
	// Validate checks the candidate against the active snapshot. It builds
	// nothing.
	Validate(ctx context.Context, candidate *Candidate) error
	// Warm builds and warms everything the candidate serves with. Nothing it
	// builds serves until the result is activated.
	Warm(ctx context.Context, candidate *Candidate) (Warmed, error)
}

// Warmed is a candidate ready to serve.
type Warmed interface {
	// Activate makes the candidate serve in one atomic step; the snapshot it
	// replaces then drains on its own. On error the active snapshot keeps
	// serving and Activate has released what Warm built.
	Activate(ctx context.Context) error
	// Discard releases what Warm built for a candidate that will not serve.
	Discard()
}

// Reporter publishes the progress of every update, for status reads.
type Reporter interface {
	ReportConfigAttempt(attempt Attempt, err error)
}

// AttemptStatus is where an update's attempt stands.
type AttemptStatus string

const (
	AttemptPreparing  AttemptStatus = "preparing"
	AttemptActive     AttemptStatus = "active"
	AttemptFailed     AttemptStatus = "failed"
	AttemptSuperseded AttemptStatus = "superseded"
)

// Attempt is one update's way through the lifecycle.
type Attempt struct {
	ID     uint64        `json:"attempt"`
	Hash   string        `json:"document_hash"`
	Origin Origin        `json:"origin"`
	Status AttemptStatus `json:"status"`
	// Stage is the lifecycle stage the attempt reached.
	Stage Stage `json:"stage"`
	// Step is the runtime's progress within the stage.
	Step string `json:"step,omitempty"`
	// Version is the version the attempt activated.
	Version uint64 `json:"version,omitempty"`
	// Reused lists the components the attempt kept from the active snapshot
	// instead of building them again.
	Reused []Component `json:"reused,omitempty"`
	// Reasons explain a rejected attempt.
	Reasons    []Reason   `json:"reasons,omitempty"`
	StartedAt  time.Time  `json:"started_at"`
	FinishedAt *time.Time `json:"finished_at,omitempty"`
}

// Status is the lifecycle's state for status reads.
type Status struct {
	// Active describes the snapshot that serves.
	Active *SnapshotInfo `json:"active,omitempty"`
	// Latest is the most recent attempt, finished or not.
	Latest *Attempt `json:"latest,omitempty"`
	// LastNACK is the most recent rejected attempt.
	LastNACK *Attempt `json:"last_nack,omitempty"`
}

// SnapshotInfo describes a snapshot without its document.
type SnapshotInfo struct {
	Version     uint64       `json:"version"`
	Hash        string       `json:"hash"`
	Origin      Origin       `json:"origin"`
	CreatedAt   time.Time    `json:"created_at"`
	ActivatedAt time.Time    `json:"activated_at"`
	Resources   map[Kind]int `json:"resources"`
}

// Options configure a Manager.
type Options struct {
	Runtime  Runtime
	Reporter Reporter
	// Parts build the components each snapshot owns besides the routing
	// pipeline, in order.
	Parts []PartBuilder
	// History records every activation; nil keeps DefaultHistoryLimit
	// records in memory.
	History *History
	// Now returns the current time; nil uses time.Now.
	Now func() time.Time
}

// attributionTTL bounds how long an attribution waits for its update.
const attributionTTL = 10 * time.Minute

// Manager is the one owner of the Router's configuration lifecycle. It runs
// updates one at a time; status reads never wait for an update in progress.
type Manager struct {
	runtime  Runtime
	reporter Reporter
	parts    []PartBuilder
	history  *History
	now      func() time.Time

	// updates serializes updates. It is held through model preparation, so
	// nothing on a request path takes it.
	updates sync.Mutex

	mu           sync.RWMutex
	active       *Snapshot
	activatedAt  time.Time
	latest       *Attempt
	lastNACK     *Attempt
	attempts     uint64
	attributions map[string]attribution
}

type attribution struct {
	origin Origin
	at     time.Time
}

// NewManager returns a Manager with no active snapshot.
func NewManager(opts Options) *Manager {
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	history := opts.History
	if history == nil {
		history, _ = NewHistory(DefaultHistoryLimit, nil)
	}
	return &Manager{
		runtime: opts.Runtime, reporter: opts.Reporter, parts: append([]PartBuilder(nil), opts.Parts...),
		history: history, now: now, attributions: make(map[string]attribution),
	}
}

// History is the record of the snapshots this Router activated.
func (m *Manager) History() *History { return m.history }

// Install compiles the configuration the Router started with, builds its
// parts and records it as the active snapshot. The Router has built its
// routing pipeline already, so the runtime is not called. A Router that
// restarts on the newest recorded document keeps its version; any other
// document takes the next one. A rejection means the Router cannot start.
func (m *Manager) Install(ctx context.Context, u Update) (*Snapshot, error) {
	m.updates.Lock()
	defer m.updates.Unlock()
	a := m.begin(u)
	snapshot, err := m.install(ctx, a, u)
	if err == nil {
		write := m.history.Begin()
		var unchanged bool
		snapshot.version, unchanged = m.installedVersion(a.attempt.Hash)
		m.activate(write, snapshot, !unchanged)
		write.End()
	}
	m.finish(a, snapshot, err)
	return snapshot, err
}

func (m *Manager) install(ctx context.Context, a *attemptState, u Update) (*Snapshot, error) {
	snapshot, err := m.compile(a, u, 0)
	if err != nil {
		return nil, err
	}
	candidate := &Candidate{snapshot: snapshot, attempt: a}
	a.enter(StageValidate)
	if err := m.validateParts(candidate); err != nil {
		return nil, err
	}
	a.enter(StageWarm)
	if err := buildParts(ctx, m.parts, candidate); err != nil {
		return nil, classify(StageWarm, err)
	}
	return snapshot, nil
}

// installedVersion is the version a starting Router serves the document with
// hash under: the newest recorded activation's when that is the document, so
// a restart keeps it, or the next one.
func (m *Manager) installedVersion(hash string) (uint64, bool) {
	if latest, ok := m.history.Latest(); ok && latest.Hash != "" && latest.Hash == hash {
		return latest.Version, true
	}
	return m.nextVersion(), false
}

// validateParts runs every part builder's check of the candidate.
func (m *Manager) validateParts(candidate *Candidate) error {
	for _, builder := range m.parts {
		if builder.Validate == nil {
			continue
		}
		if err := builder.Validate(candidate.snapshot, candidate.active); err != nil {
			return classify(StageValidate, err)
		}
	}
	return nil
}

// Apply runs one update through compile, validate, warm and activate. It
// returns the activated snapshot, or the error that stopped the update: a
// *Rejection (the active snapshot keeps serving), ErrSuperseded, or the
// context's error.
func (m *Manager) Apply(ctx context.Context, u Update) (*Snapshot, error) {
	m.updates.Lock()
	defer m.updates.Unlock()
	u = m.attributed(u)
	a := m.begin(u)
	snapshot, err := m.run(ctx, a, u)
	m.finish(a, snapshot, err)
	return snapshot, err
}

// Attribute names who causes the next file update of the document with hash,
// such as the management API writing the watched file. That update takes
// origin instead of the file source's. withdraw drops an attribution no
// update used, for a write that failed.
func (m *Manager) Attribute(hash string, origin Origin) (withdraw func()) {
	if hash == "" {
		return func() {}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	now := m.now()
	for key, pending := range m.attributions {
		if now.Sub(pending.at) > attributionTTL {
			delete(m.attributions, key)
		}
	}
	m.attributions[hash] = attribution{origin: origin, at: now}
	return func() {
		m.mu.Lock()
		defer m.mu.Unlock()
		if pending, ok := m.attributions[hash]; ok && pending.at.Equal(now) {
			delete(m.attributions, hash)
		}
	}
}

// attributed is u with the origin attributed to its document, if any.
func (m *Manager) attributed(u Update) Update {
	if u.Origin.Source != SourceFile {
		return u
	}
	hash := documentHash(u.Config, u.Document)
	m.mu.Lock()
	defer m.mu.Unlock()
	if pending, ok := m.attributions[hash]; ok && hash != "" {
		delete(m.attributions, hash)
		u.Origin = pending.origin
	}
	return u
}

// nextVersion is the version the next activation takes: one past every
// version this Router activated, before a restart included.
func (m *Manager) nextVersion() uint64 {
	var last uint64
	if latest, ok := m.history.Latest(); ok {
		last = latest.Version
	}
	m.mu.RLock()
	if m.active != nil && m.active.version > last {
		last = m.active.version
	}
	m.mu.RUnlock()
	return last + 1
}

// Reject records an update that its source rejected before handing it over,
// such as a document that does not parse. err keeps its message; a stage the
// source did not classify is StageParse. An error the lifecycle itself
// returned is already recorded and is returned unchanged.
func (m *Manager) Reject(u Update, err error) error {
	var rejection *Rejection
	if err == nil || errors.Is(err, ErrSuperseded) || (errors.As(err, &rejection) && rejection.recorded) {
		return err
	}
	m.updates.Lock()
	defer m.updates.Unlock()
	a := m.begin(u)
	classified := classify(StageParse, err)
	if errors.As(classified, &rejection) {
		a.step(string(rejection.Stage))
	}
	m.finish(a, nil, classified)
	return classified
}

// Active returns the snapshot that serves, or nil before the first one.
func (m *Manager) Active() *Snapshot {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.active
}

// Status returns the lifecycle's state.
func (m *Manager) Status() Status {
	m.mu.RLock()
	defer m.mu.RUnlock()
	status := Status{Latest: cloneAttempt(m.latest), LastNACK: cloneAttempt(m.lastNACK)}
	if m.active != nil {
		status.Active = &SnapshotInfo{
			Version: m.active.version, Hash: m.active.hash, Origin: m.active.origin,
			CreatedAt: m.active.createdAt, ActivatedAt: m.activatedAt, Resources: m.active.resources.Counts(),
		}
	}
	return status
}

func (m *Manager) run(ctx context.Context, a *attemptState, u Update) (*Snapshot, error) {
	if err := ctx.Err(); err != nil {
		return nil, Reject(StageCompile, CodeCanceled, err)
	}
	snapshot, err := m.compile(a, u, 0)
	if err != nil {
		return nil, err
	}
	if m.runtime == nil {
		return nil, Reject(StageValidate, CodeUnsupported, errors.New("the configuration lifecycle has no runtime"))
	}
	candidate := &Candidate{snapshot: snapshot, active: m.Active(), current: u.Current, attempt: a}
	a.enter(StageValidate)
	if validateErr := m.runtime.Validate(ctx, candidate); validateErr != nil {
		return nil, classify(StageValidate, validateErr)
	}
	if validateErr := m.validateParts(candidate); validateErr != nil {
		return nil, validateErr
	}
	a.enter(StageWarm)
	if partsErr := buildParts(ctx, m.parts, candidate); partsErr != nil {
		return nil, classify(StageWarm, partsErr)
	}
	warmed, err := m.runtime.Warm(ctx, candidate)
	if err != nil {
		return nil, withRelease(classify(StageWarm, err), snapshot)
	}
	a.enter(StageActivate)
	// The version is taken while the history is held, so writers of one
	// document never activate the same version.
	write := m.history.Begin()
	defer write.End()
	snapshot.version = m.nextVersion()
	if activateErr := warmed.Activate(ctx); activateErr != nil {
		return nil, withRelease(classify(StageActivate, activateErr), snapshot)
	}
	m.activate(write, snapshot, true)
	return snapshot, nil
}

// withRelease releases the parts of a candidate that will not serve and adds
// a failure to do so to err.
func withRelease(err error, candidate *Snapshot) error {
	if releaseErr := candidate.Release(context.Background()); releaseErr != nil {
		return errors.Join(err, releaseErr)
	}
	return err
}

// compile builds the candidate snapshot under version.
func (m *Manager) compile(a *attemptState, u Update, version uint64) (*Snapshot, error) {
	a.enter(StageCompile)
	if u.Config == nil {
		return nil, Reject(StageParse, CodeInvalidDocument, errors.New("config reload candidate is nil"))
	}
	resources, err := Compile(u.Config)
	if err != nil {
		return nil, err
	}
	return &Snapshot{
		version: version, hash: a.attempt.Hash, origin: u.Origin, document: u.Document, config: u.Config,
		resources: resources, createdAt: m.now().UTC(),
	}, nil
}

// activate makes snapshot the active one and, when record is set, adds it to
// the history within write. The history is best effort: a store that cannot
// persist the record keeps the Router serving the activated snapshot.
func (m *Manager) activate(write *HistoryWrite, snapshot *Snapshot, record bool) {
	activatedAt := m.now().UTC()
	m.mu.Lock()
	m.active, m.activatedAt = snapshot, activatedAt
	m.mu.Unlock()
	if !record {
		return
	}
	if _, err := write.Add(Record{
		Version: snapshot.version, Hash: snapshot.hash, Origin: snapshot.origin, RecordedAt: activatedAt,
		Document: snapshot.document,
	}); err != nil {
		logging.ComponentWarnEvent("config", "config_history_record_failed", map[string]interface{}{
			"version": snapshot.version,
			"error":   err.Error(),
		})
	}
}

// classify turns err into the stage's rejection, unless it is not one: a
// superseded update, or a rejection a later stage already classified.
func classify(stage Stage, err error) error {
	if errors.Is(err, ErrSuperseded) {
		return err
	}
	var rejection *Rejection
	if errors.As(err, &rejection) {
		return err
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return Reject(stage, CodeCanceled, err)
	}
	return Reject(stage, defaultCode[stage], err)
}

var defaultCode = map[Stage]Code{
	StageParse:    CodeInvalidDocument,
	StageCompile:  CodeInvalidResource,
	StageValidate: CodeInvalidResource,
	StageWarm:     CodeBuildFailed,
	StageActivate: CodeActivationFailed,
}

// attemptState is the mutable record of the attempt in progress; status
// reads see copies of it.
type attemptState struct {
	m       *Manager
	attempt Attempt
}

func (m *Manager) begin(u Update) *attemptState {
	m.mu.Lock()
	m.attempts++
	a := &attemptState{m: m, attempt: Attempt{
		ID: m.attempts, Hash: documentHash(u.Config, u.Document), Origin: u.Origin,
		Status: AttemptPreparing, Stage: StageCompile, Step: "validation", StartedAt: m.now().UTC(),
	}}
	m.latest = cloneAttempt(&a.attempt)
	m.mu.Unlock()
	m.report(a.attempt, nil)
	return a
}

func (a *attemptState) enter(stage Stage) {
	a.update(func(attempt *Attempt) { attempt.Stage = stage })
}

func (a *attemptState) step(step string) {
	a.update(func(attempt *Attempt) { attempt.Step = step })
}

func (a *attemptState) update(change func(*Attempt)) {
	a.m.mu.Lock()
	change(&a.attempt)
	a.m.latest = cloneAttempt(&a.attempt)
	snapshot := a.attempt
	a.m.mu.Unlock()
	a.m.report(snapshot, nil)
}

func (m *Manager) finish(a *attemptState, snapshot *Snapshot, err error) {
	finished := m.now().UTC()
	m.mu.Lock()
	a.attempt.FinishedAt = &finished
	switch {
	case err == nil:
		a.attempt.Status = AttemptActive
		a.attempt.Version = snapshot.version
	case errors.Is(err, ErrSuperseded):
		a.attempt.Status = AttemptSuperseded
	default:
		a.attempt.Status = AttemptFailed
		var rejection *Rejection
		if errors.As(err, &rejection) {
			a.attempt.Stage = rejection.Stage
			rejection.recorded = true
		}
		a.attempt.Reasons = ReasonsOf(err)
		m.lastNACK = cloneAttempt(&a.attempt)
	}
	m.latest = cloneAttempt(&a.attempt)
	final := a.attempt
	m.mu.Unlock()
	m.report(final, err)
	metrics.RecordConfigUpdate(string(final.Origin.Source), string(final.Status), string(final.Stage), finished.Sub(final.StartedAt))
	switch final.Status {
	case AttemptActive:
		metrics.SetConfigActive(final.Version, final.Hash)
	case AttemptFailed:
		metrics.SetConfigLastRejection(finished)
	}
}

func (m *Manager) report(attempt Attempt, err error) {
	if m.reporter != nil {
		attempt.Reasons = append([]Reason(nil), attempt.Reasons...)
		m.reporter.ReportConfigAttempt(attempt, err)
	}
}

func cloneAttempt(a *Attempt) *Attempt {
	if a == nil {
		return nil
	}
	clone := *a
	clone.Reasons = append([]Reason(nil), a.Reasons...)
	clone.Reused = append([]Component(nil), a.Reused...)
	if a.FinishedAt != nil {
		finished := *a.FinishedAt
		clone.FinishedAt = &finished
	}
	return &clone
}

// Candidate is a snapshot on its way through the lifecycle.
type Candidate struct {
	snapshot *Snapshot
	active   *Snapshot
	current  func() error
	attempt  *attemptState
}

// Snapshot is the candidate snapshot.
func (c *Candidate) Snapshot() *Snapshot { return c.snapshot }

// Active is the snapshot that serves while the candidate is prepared, or nil
// before the first activation.
func (c *Candidate) Active() *Snapshot { return c.active }

// Current reports whether the candidate is still its source's latest
// document: nil, or ErrSuperseded.
func (c *Candidate) Current() error {
	if c.current == nil {
		return nil
	}
	return c.current()
}

// Step reports the runtime's progress within the current stage.
func (c *Candidate) Step(step string) {
	if c.attempt != nil {
		c.attempt.step(step)
	}
}

// RecordReuse notes that the candidate keeps component from the active
// snapshot, the same instance, instead of building it again.
func (c *Candidate) RecordReuse(component Component) {
	c.snapshot.parts.reused = append(c.snapshot.parts.reused, component)
	if c.attempt != nil {
		c.attempt.update(func(attempt *Attempt) { attempt.Reused = append(attempt.Reused, component) })
	}
}
