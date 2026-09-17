package routerruntime

import "time"

// ConfigActivation describes the most recent attempt to prepare and publish a
// configuration. It belongs to the Router registry, not the API or file watcher.
// DocumentHash identifies the exact candidate; an older attempt cannot update
// the status of a newer document.
type ConfigActivation struct {
	Attempt      uint64     `json:"attempt"`
	DocumentHash string     `json:"document_hash"`
	Source       string     `json:"source"`
	Status       string     `json:"status"`
	Stage        string     `json:"stage"`
	StartedAt    time.Time  `json:"started_at"`
	FinishedAt   *time.Time `json:"finished_at,omitempty"`
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

func (r *Registry) ConfigActivation() ConfigActivation {
	if r == nil {
		return ConfigActivation{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := r.configActivation
	if result.FinishedAt != nil {
		finished := *result.FinishedAt
		result.FinishedAt = &finished
	}
	return result
}
