package retention

import (
	"context"
	"errors"
	"strconv"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type fakeAdapter struct {
	capability Capability
	ackErr     error
	observeErr error
}

func (f fakeAdapter) Capability() Capability { return f.capability }

func (f fakeAdapter) Translate(d Directive) Translation {
	actions := make([]Action, 0, 4)
	if d.Drop != nil && f.capability.Supports(FeatureDrop) {
		actions = append(actions, Action{Feature: FeatureDrop, Value: boolString(*d.Drop)})
	}
	if d.TTLTurns != nil && f.capability.Supports(FeatureTTLTurns) {
		actions = append(actions, Action{Feature: FeatureTTLTurns, Value: intString(*d.TTLTurns)})
	}
	if d.KeepCurrentModel != nil && f.capability.Supports(FeatureKeepCurrentModel) {
		actions = append(actions, Action{Feature: FeatureKeepCurrentModel, Value: boolString(*d.KeepCurrentModel)})
	}
	if d.PreferPrefixRetention != nil && f.capability.Supports(FeaturePreferPrefixRetention) {
		actions = append(actions, Action{Feature: FeaturePreferPrefixRetention, Value: boolString(*d.PreferPrefixRetention)})
	}
	if len(actions) == 0 {
		return Translation{Capability: f.capability, Status: StatusUnsupported, Reason: "no_supported_retention_feature"}
	}
	return Translation{Capability: f.capability, Actions: actions, Status: StatusTranslated}
}

func (f fakeAdapter) Acknowledge(_ context.Context, translation Translation) (Acknowledgement, error) {
	if f.ackErr != nil {
		return Acknowledgement{}, f.ackErr
	}
	return Acknowledgement{Capability: translation.Capability, Actions: translation.Actions, Status: StatusAcknowledged}, nil
}

func (f fakeAdapter) Observe(_ context.Context, acknowledgement Acknowledgement) (Observation, error) {
	if f.observeErr != nil {
		return Observation{}, f.observeErr
	}
	return Observation{Status: StatusObserved, Enforced: true, Reason: string(acknowledgement.Status)}, nil
}

func TestResolveWithoutAdapterIsExplicitNoOp(t *testing.T) {
	trueValue := true
	outcome := Resolve(context.Background(), &config.RetentionDirective{Drop: &trueValue}, nil)
	if outcome.SchemaVersion != SchemaVersion || outcome.Status != StatusUnsupported {
		t.Fatalf("outcome = %+v, want versioned unsupported result", outcome)
	}
	if outcome.Reason != "no_retention_adapter" || outcome.Requested == nil || outcome.Requested.Drop == nil || !*outcome.Requested.Drop {
		t.Fatalf("unsupported outcome lost requested directive: %+v", outcome)
	}
}

func TestResolveTracksTranslationAcknowledgementAndObservation(t *testing.T) {
	turns := 2
	adapter := fakeAdapter{capability: Capability{
		Backend: "fake", Version: "v1", Features: []Feature{FeatureTTLTurns},
	}}
	outcome := Resolve(context.Background(), &config.RetentionDirective{TTLTurns: &turns}, adapter)
	if outcome.Status != StatusObserved || outcome.Translated == nil || outcome.Acknowledged == nil || outcome.Observed == nil {
		t.Fatalf("outcome = %+v, want complete lifecycle receipt", outcome)
	}
	if len(outcome.Translated.Actions) != 1 || outcome.Translated.Actions[0].Value != "2" || !outcome.Observed.Enforced {
		t.Fatalf("unexpected receipt: %+v", outcome)
	}
}

func TestResolveUnsupportedFeatureDoesNotAcknowledge(t *testing.T) {
	trueValue := true
	adapter := fakeAdapter{capability: Capability{Backend: "fake", Version: "v1", Features: []Feature{FeatureTTLTurns}}}
	outcome := Resolve(context.Background(), &config.RetentionDirective{Drop: &trueValue}, adapter)
	if outcome.Status != StatusUnsupported || outcome.Acknowledged != nil || outcome.Observed != nil {
		t.Fatalf("outcome = %+v, want unsupported without fabricated acknowledgement", outcome)
	}
}

func TestResolveAcknowledgementFailureIsNotEnforcement(t *testing.T) {
	trueValue := true
	adapter := fakeAdapter{capability: Capability{Backend: "fake", Version: "v1", Features: []Feature{FeatureDrop}}, ackErr: errors.New("offline")}
	outcome := Resolve(context.Background(), &config.RetentionDirective{Drop: &trueValue}, adapter)
	if outcome.Status != StatusFailed || outcome.Acknowledged != nil || outcome.Observed != nil {
		t.Fatalf("outcome = %+v, want failed without acknowledgement", outcome)
	}
}

func TestOutcomeCloneDoesNotAliasLifecycleSlices(t *testing.T) {
	turns := 1
	outcome := Resolve(context.Background(), &config.RetentionDirective{TTLTurns: &turns}, fakeAdapter{capability: Capability{Backend: "fake", Version: "v1", Features: []Feature{FeatureTTLTurns}}})
	clone := outcome.Clone()
	clone.Translated.Actions[0].Value = "9"
	clone.Requested.TTLTurns = &turns
	*clone.Requested.TTLTurns = 9
	if outcome.Translated.Actions[0].Value != "1" || *outcome.Requested.TTLTurns != 1 {
		t.Fatalf("clone mutation changed original outcome: %+v", outcome)
	}
}

func boolString(value bool) string {
	if value {
		return "true"
	}
	return "false"
}

func intString(value int) string {
	return strconv.Itoa(value)
}
