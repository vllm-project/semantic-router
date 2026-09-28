package routerruntime

import (
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

const StatusSchemaVersion = "v1"

type ConditionStatus string

const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"
)

type ConditionType string

const (
	ConditionLive            ConditionType = "Live"
	ConditionStartupComplete ConditionType = "StartupComplete"
	ConditionActiveConfig    ConditionType = "ActiveConfigAvailable"
	ConditionConverged       ConditionType = "ConfigConverged"
	ConditionReady           ConditionType = "Ready"
)

type StatusReason string

const (
	ReasonProcessRunning                 StatusReason = "ProcessRunning"
	ReasonStartupUnobserved              StatusReason = "StartupUnobserved"
	ReasonStartupIncomplete              StatusReason = "StartupIncomplete"
	ReasonStartupFailed                  StatusReason = "StartupFailed"
	ReasonStartupComplete                StatusReason = "StartupComplete"
	ReasonNoActiveConfig                 StatusReason = "NoActiveConfig"
	ReasonActiveConfigAvailable          StatusReason = "ActiveConfigAvailable"
	ReasonConfigIdentityUnobserved       StatusReason = "ConfigIdentityUnobserved"
	ReasonConfigHashesMatch              StatusReason = "ConfigHashesMatch"
	ReasonConfigActivationPending        StatusReason = "ConfigActivationPending"
	ReasonConfigActivationFailed         StatusReason = "ConfigActivationFailed"
	ReasonConfigActivationSuperseded     StatusReason = "ConfigActivationSuperseded"
	ReasonRequiredDependenciesUnobserved StatusReason = "RequiredDependenciesUnobserved"
)

type StatusCondition struct {
	Type   ConditionType   `json:"type"`
	Status ConditionStatus `json:"status"`
	Reason StatusReason    `json:"reason"`
}

type StartupObservation struct {
	Phase      string    `json:"phase"`
	Ready      bool      `json:"ready"`
	ObservedAt time.Time `json:"observed_at"`
}

type ConfigStatus struct {
	ActiveHash       string `json:"active_hash,omitempty"`
	ObservedHash     string `json:"observed_hash,omitempty"`
	ObservedAttempt  uint64 `json:"observed_attempt"`
	ActivationStatus string `json:"activation_status,omitempty"`
	ActivationStage  string `json:"activation_stage,omitempty"`
}

// StatusReport contains only replica-local observations, not fleet or probe state.
type StatusReport struct {
	SchemaVersion string              `json:"schema_version"`
	Scope         string              `json:"scope"`
	InstanceID    string              `json:"instance_id"`
	ObservedAt    time.Time           `json:"observed_at"`
	Startup       *StartupObservation `json:"startup,omitempty"`
	Config        ConfigStatus        `json:"config"`
	Conditions    []StatusCondition   `json:"conditions"`
}

type localStartupWriter struct {
	registry *Registry
	writer   startupstatus.StatusWriter
}

type localStartupSnapshot struct {
	state      startupstatus.State
	observedAt time.Time
}

func cloneStartupState(state startupstatus.State) startupstatus.State {
	state.PendingModels = slices.Clone(state.PendingModels)
	if state.EmbeddingProvider != nil {
		provider := *state.EmbeddingProvider
		if provider.APIKeyEnvSet != nil {
			value := *provider.APIKeyEnvSet
			provider.APIKeyEnvSet = &value
		}
		if provider.Healthy != nil {
			value := *provider.Healthy
			provider.Healthy = &value
		}
		state.EmbeddingProvider = &provider
	}
	return state
}

// StartupStatusWriter retains local facts even when their external store is unavailable.
func (r *Registry) StartupStatusWriter(writer startupstatus.StatusWriter) startupstatus.StatusWriter {
	return &localStartupWriter{registry: r, writer: writer}
}

func (w *localStartupWriter) Write(state startupstatus.State) error {
	snapshot := cloneStartupState(state)
	w.registry.mu.Lock()
	observedAt := time.Now().UTC()
	snapshot.UpdatedAt = observedAt.Format(time.RFC3339)
	w.registry.startupStatus = &localStartupSnapshot{state: snapshot, observedAt: observedAt}
	w.registry.mu.Unlock()
	return w.writer.Write(state)
}

func (r *Registry) StartupState() *startupstatus.State {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.startupStatus == nil {
		return nil
	}
	state := cloneStartupState(r.startupStatus.state)
	return &state
}

// Status returns an atomic observation of this registry without reading shared storage.
func (r *Registry) Status() StatusReport {
	r.mu.RLock()
	report := StatusReport{
		SchemaVersion: StatusSchemaVersion,
		Scope:         "replica",
		InstanceID:    r.instanceID,
		ObservedAt:    time.Now().UTC(),
	}
	if r.startupStatus != nil {
		report.Startup = &StartupObservation{
			Phase:      r.startupStatus.state.Phase,
			Ready:      r.startupStatus.state.Ready,
			ObservedAt: r.startupStatus.observedAt,
		}
	}
	active := r.config != nil && r.classificationService != nil
	if active && r.acquireGeneration != nil {
		release, acquired := r.acquireGeneration()
		active = acquired
		if acquired {
			defer release()
		}
	}
	if r.config != nil {
		report.Config.ObservedHash = r.config.DocumentHash
		if active {
			report.Config.ActiveHash = r.config.DocumentHash
		}
	}
	if r.configActivation.Attempt != 0 {
		report.Config.ObservedHash = r.configActivation.DocumentHash
		report.Config.ObservedAttempt = r.configActivation.Attempt
		report.Config.ActivationStatus = r.configActivation.Status
		report.Config.ActivationStage = r.configActivation.Stage
	}
	r.mu.RUnlock()

	startup := StatusCondition{Type: ConditionStartupComplete, Status: ConditionUnknown, Reason: ReasonStartupUnobserved}
	if report.Startup != nil {
		startup.Status, startup.Reason = ConditionFalse, ReasonStartupIncomplete
		if report.Startup.Ready {
			startup.Status, startup.Reason = ConditionTrue, ReasonStartupComplete
		} else if report.Startup.Phase == "error" {
			startup.Reason = ReasonStartupFailed
		}
	}
	activeConfig := StatusCondition{Type: ConditionActiveConfig, Status: ConditionFalse, Reason: ReasonNoActiveConfig}
	if active {
		activeConfig.Status, activeConfig.Reason = ConditionTrue, ReasonActiveConfigAvailable
	}
	converged := StatusCondition{Type: ConditionConverged, Status: ConditionUnknown, Reason: ReasonConfigIdentityUnobserved}
	if report.Config.ActiveHash != "" && report.Config.ObservedHash != "" {
		converged.Status, converged.Reason = ConditionFalse, ReasonConfigActivationPending
		switch {
		case report.Config.ActiveHash == report.Config.ObservedHash:
			converged.Status, converged.Reason = ConditionTrue, ReasonConfigHashesMatch
		case report.Config.ActivationStatus == "failed":
			converged.Reason = ReasonConfigActivationFailed
		case report.Config.ActivationStatus == "superseded":
			converged.Reason = ReasonConfigActivationSuperseded
		}
	}
	ready := StatusCondition{Type: ConditionReady, Status: ConditionUnknown, Reason: ReasonRequiredDependenciesUnobserved}
	if startup.Status != ConditionTrue {
		ready.Status, ready.Reason = startup.Status, startup.Reason
	}
	if !active {
		ready.Status, ready.Reason = ConditionFalse, ReasonNoActiveConfig
	}
	report.Conditions = []StatusCondition{
		{Type: ConditionLive, Status: ConditionTrue, Reason: ReasonProcessRunning},
		startup,
		activeConfig,
		converged,
		ready,
	}
	return report
}
