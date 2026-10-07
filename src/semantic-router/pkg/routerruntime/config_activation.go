package routerruntime

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
)

// ConfigActivation describes the most recent attempt to prepare and publish a
// configuration. The configuration lifecycle reports it; the registry only
// publishes it for the API and status reads. DocumentHash identifies the exact
// candidate; an older attempt cannot update the status of a newer document.
type ConfigActivation struct {
	Attempt      uint64     `json:"attempt"`
	DocumentHash string     `json:"document_hash"`
	Source       string     `json:"source"`
	Status       string     `json:"status"`
	Stage        string     `json:"stage"`
	StartedAt    time.Time  `json:"started_at"`
	FinishedAt   *time.Time `json:"finished_at,omitempty"`
	// Version is the configuration snapshot version an active attempt
	// activated.
	Version uint64 `json:"version,omitempty"`
	// Reasons explain why a failed attempt was rejected.
	Reasons []configsnapshot.Reason `json:"reasons,omitempty"`
	// The management boundary redacts this diagnostic before serialization.
	FailureDetail string `json:"-"`
}

func (r *Registry) BeginConfigActivation(hash, source string) uint64 {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	attempt := r.configActivation.Attempt + 1
	r.configActivation = ConfigActivation{
		Attempt: attempt, DocumentHash: hash, Source: source,
		Status: "preparing", Stage: "validation", StartedAt: time.Now().UTC(),
	}
	return attempt
}

func (r *Registry) SetConfigActivationStage(attempt uint64, stage string) {
	if r == nil || attempt == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.configActivation.Attempt == attempt && r.configActivation.Status == "preparing" {
		r.configActivation.Stage = stage
	}
}

func (r *Registry) FinishConfigActivation(attempt uint64, status string, err error) {
	if r == nil || attempt == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.configActivation.Attempt != attempt || r.configActivation.Status != "preparing" {
		return
	}
	finished := time.Now().UTC()
	r.configActivation.Status = status
	r.configActivation.FinishedAt = &finished
	if err != nil {
		r.configActivation.FailureDetail = err.Error()
	}
}

// ReportConfigAttempt publishes the lifecycle's latest attempt. An attempt
// older than the published one never replaces it.
func (r *Registry) ReportConfigAttempt(attempt configsnapshot.Attempt, err error) {
	if r == nil {
		return
	}
	activation := ConfigActivation{
		Attempt: attempt.ID, DocumentHash: attempt.Hash, Source: string(attempt.Origin.Source),
		Status: string(attempt.Status), Stage: attempt.Step, StartedAt: attempt.StartedAt,
		Version: attempt.Version, Reasons: attempt.Reasons,
	}
	if attempt.FinishedAt != nil {
		finished := *attempt.FinishedAt
		activation.FinishedAt = &finished
	}
	if err != nil {
		activation.FailureDetail = err.Error()
	}
	r.mu.Lock()
	if activation.Attempt < r.configActivation.Attempt {
		r.mu.Unlock()
		return
	}
	r.configActivation = activation
	if attempt.Status == configsnapshot.AttemptFailed {
		rejected := activation
		r.configRejection = &rejected
	}
	listeners := append([]func(configsnapshot.Attempt){}, r.configAttemptListeners...)
	r.mu.Unlock()
	if attempt.FinishedAt == nil {
		return
	}
	for _, listener := range listeners {
		listener(attempt)
	}
}

// OnConfigAttempt registers a listener that receives every configuration
// attempt once it has finished.
func (r *Registry) OnConfigAttempt(listener func(configsnapshot.Attempt)) {
	if r == nil || listener == nil {
		return
	}
	r.mu.Lock()
	r.configAttemptListeners = append(r.configAttemptListeners, listener)
	r.mu.Unlock()
}

func (r *Registry) ConfigActivation() ConfigActivation {
	if r == nil {
		return ConfigActivation{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return cloneConfigActivation(r.configActivation)
}

// LastConfigRejection returns the most recent rejected attempt, if any.
func (r *Registry) LastConfigRejection() (ConfigActivation, bool) {
	if r == nil {
		return ConfigActivation{}, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.configRejection == nil {
		return ConfigActivation{}, false
	}
	return cloneConfigActivation(*r.configRejection), true
}

// SetConfigLifecycle publishes the Router's configuration lifecycle, which
// owns the configuration history and attributes management changes.
func (r *Registry) SetConfigLifecycle(lifecycle *configsnapshot.Manager) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.configLifecycle = lifecycle
	r.mu.Unlock()
}

// ConfigLifecycle returns the published configuration lifecycle, or nil.
func (r *Registry) ConfigLifecycle() *configsnapshot.Manager {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.configLifecycle
}

// ConfigSnapshot returns the configuration snapshot of the published router
// generation, or nil when the generation was published without one.
func (r *Registry) ConfigSnapshot() *configsnapshot.Snapshot {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.configSnapshot
}

func cloneConfigActivation(activation ConfigActivation) ConfigActivation {
	if activation.FinishedAt != nil {
		finished := *activation.FinishedAt
		activation.FinishedAt = &finished
	}
	activation.Reasons = append([]configsnapshot.Reason(nil), activation.Reasons...)
	return activation
}
