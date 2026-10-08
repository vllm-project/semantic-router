// Package retention defines the provider-neutral contract for backend cache
// retention acknowledgements. It deliberately does not depend on a backend
// client or promise that a backend applied a directive.
package retention

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// SchemaVersion identifies the versioned retention bridge contract.
const SchemaVersion = "semantic-router.retention/v1"

// Feature is a capability that a backend adapter can explicitly advertise.
type Feature string

const (
	FeatureDrop                  Feature = "drop"
	FeatureTTLTurns              Feature = "ttl_turns"
	FeatureKeepCurrentModel      Feature = "keep_current_model"
	FeaturePreferPrefixRetention Feature = "prefer_prefix_retention"
)

// Status is the lifecycle state of a retention bridge result.
type Status string

const (
	StatusNotRequested Status = "not_requested"
	StatusTranslated   Status = "translated"
	StatusAcknowledged Status = "acknowledged"
	StatusObserved     Status = "observed"
	StatusUnsupported  Status = "unsupported"
	StatusFailed       Status = "failed"
)

// Directive is an immutable, tri-state snapshot of config.RetentionDirective.
// Pointer fields preserve the difference between an unset value and an
// explicit false or zero.
type Directive struct {
	Drop                  *bool `json:"drop,omitempty"`
	TTLTurns              *int  `json:"ttl_turns,omitempty"`
	KeepCurrentModel      *bool `json:"keep_current_model,omitempty"`
	PreferPrefixRetention *bool `json:"prefer_prefix_retention,omitempty"`
}

// FromConfig copies the existing Router directive into the bridge contract.
func FromConfig(value *config.RetentionDirective) *Directive {
	if value == nil {
		return nil
	}
	return &Directive{
		Drop:                  cloneBool(value.Drop),
		TTLTurns:              cloneInt(value.TTLTurns),
		KeepCurrentModel:      cloneBool(value.KeepCurrentModel),
		PreferPrefixRetention: cloneBool(value.PreferPrefixRetention),
	}
}

// Capability describes one version-pinned backend adapter capability.
type Capability struct {
	Backend  string    `json:"backend"`
	Version  string    `json:"version"`
	Features []Feature `json:"features,omitempty"`
}

func (c Capability) Supports(feature Feature) bool {
	for _, candidate := range c.Features {
		if candidate == feature {
			return true
		}
	}
	return false
}

// Action is the provider-neutral translated operation. Parameters contain
// only bounded retention facts and never prompt, cache-key, or tenant data.
type Action struct {
	Feature Feature `json:"feature"`
	Value   string  `json:"value,omitempty"`
}

// Translation is the adapter's versioned interpretation of a request.
type Translation struct {
	Capability Capability `json:"capability"`
	Actions    []Action   `json:"actions,omitempty"`
	Status     Status     `json:"status"`
	Reason     string     `json:"reason,omitempty"`
}

// Acknowledgement records what the backend adapter accepted. It is not proof
// that the backend has evicted or retained any cache entry yet.
type Acknowledgement struct {
	Capability Capability `json:"capability"`
	Actions    []Action   `json:"actions,omitempty"`
	Status     Status     `json:"status"`
	Reason     string     `json:"reason,omitempty"`
}

// Observation records a later backend observation, when a version-pinned
// adapter can prove what happened. Unsupported adapters never fabricate one.
type Observation struct {
	Status   Status `json:"status"`
	Enforced bool   `json:"enforced"`
	Reason   string `json:"reason,omitempty"`
}

// Outcome is the bounded Replay/telemetry receipt for one directive.
type Outcome struct {
	SchemaVersion string           `json:"schema_version"`
	Requested     *Directive       `json:"requested,omitempty"`
	Translated    *Translation     `json:"translated,omitempty"`
	Acknowledged  *Acknowledgement `json:"acknowledged,omitempty"`
	Observed      *Observation     `json:"observed,omitempty"`
	Status        Status           `json:"status"`
	Reason        string           `json:"reason,omitempty"`
}

// Adapter translates a directive and may acknowledge or observe it through a
// version-pinned backend integration. Implementations must not infer support
// from a successful HTTP response alone.
type Adapter interface {
	Capability() Capability
	Translate(Directive) Translation
	Acknowledge(context.Context, Translation) (Acknowledgement, error)
	Observe(context.Context, Acknowledgement) (Observation, error)
}

// Resolve evaluates one directive through an optional adapter. A nil adapter
// is a safe, explicit no-op; it never claims that retention was enforced.
func Resolve(ctx context.Context, directive *config.RetentionDirective, adapter Adapter) Outcome {
	outcome := Outcome{SchemaVersion: SchemaVersion, Requested: FromConfig(directive)}
	if directive == nil {
		outcome.Status = StatusNotRequested
		return outcome
	}
	if adapter == nil {
		outcome.Status = StatusUnsupported
		outcome.Reason = "no_retention_adapter"
		return outcome
	}
	requested := *FromConfig(directive)
	translation := adapter.Translate(requested)
	outcome.Translated = &translation
	if translation.Status == StatusUnsupported {
		outcome.Status = StatusUnsupported
		outcome.Reason = translation.Reason
		return outcome
	}
	if translation.Status != StatusTranslated {
		outcome.Status = StatusFailed
		outcome.Reason = "adapter_returned_invalid_translation_status"
		return outcome
	}
	acknowledged, err := adapter.Acknowledge(ctx, translation)
	if err != nil {
		outcome.Status = StatusFailed
		outcome.Reason = "acknowledgement_failed"
		return outcome
	}
	outcome.Acknowledged = &acknowledged
	if acknowledged.Status != StatusAcknowledged {
		outcome.Status = StatusFailed
		outcome.Reason = "adapter_returned_invalid_acknowledgement_status"
		return outcome
	}
	observed, err := adapter.Observe(ctx, acknowledged)
	if err != nil {
		outcome.Status = StatusAcknowledged
		outcome.Reason = "observation_unavailable"
		return outcome
	}
	outcome.Observed = &observed
	if observed.Status != StatusObserved {
		outcome.Status = StatusFailed
		outcome.Reason = "adapter_returned_invalid_observation_status"
		return outcome
	}
	outcome.Status = StatusObserved
	return outcome
}

// Clone returns an independent receipt for Replay storage.
func (o *Outcome) Clone() *Outcome {
	if o == nil {
		return nil
	}
	clone := *o
	clone.Requested = cloneDirective(o.Requested)
	clone.Translated = cloneTranslation(o.Translated)
	clone.Acknowledged = cloneAcknowledgement(o.Acknowledged)
	if o.Observed != nil {
		observed := *o.Observed
		clone.Observed = &observed
	}
	return &clone
}

func cloneDirective(value *Directive) *Directive {
	if value == nil {
		return nil
	}
	clone := *value
	clone.Drop = cloneBool(value.Drop)
	clone.TTLTurns = cloneInt(value.TTLTurns)
	clone.KeepCurrentModel = cloneBool(value.KeepCurrentModel)
	clone.PreferPrefixRetention = cloneBool(value.PreferPrefixRetention)
	return &clone
}

func cloneTranslation(value *Translation) *Translation {
	if value == nil {
		return nil
	}
	clone := *value
	clone.Capability.Features = append([]Feature(nil), value.Capability.Features...)
	clone.Actions = append([]Action(nil), value.Actions...)
	return &clone
}

func cloneAcknowledgement(value *Acknowledgement) *Acknowledgement {
	if value == nil {
		return nil
	}
	clone := *value
	clone.Capability.Features = append([]Feature(nil), value.Capability.Features...)
	clone.Actions = append([]Action(nil), value.Actions...)
	return &clone
}

func cloneBool(value *bool) *bool {
	if value == nil {
		return nil
	}
	clone := *value
	return &clone
}

func cloneInt(value *int) *int {
	if value == nil {
		return nil
	}
	clone := *value
	return &clone
}
