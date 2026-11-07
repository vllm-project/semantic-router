package configsnapshot

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type activatedPart struct {
	countingPart
	activations atomic.Int32
}

func (p *activatedPart) Activate() { p.activations.Add(1) }

func TestPartsPublishOnlyAfterSuccessfulActivation(t *testing.T) {
	t.Setenv("TEST_HOSTED_KEY", "hosted-secret-value")
	var built []*activatedPart
	builder := PartBuilder{
		Component: ComponentBackendTelemetry,
		Build: func(context.Context, *Snapshot, Part) (Part, error) {
			part := &activatedPart{}
			built = append(built, part)
			return part, nil
		},
	}
	runtime := &fakeRuntime{}
	manager := NewManager(Options{Runtime: runtime, Parts: []PartBuilder{builder}})
	first, err := manager.Install(t.Context(), Update{Config: parseTestConfig(t, testDocument)})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = first.Release(context.Background()) }()
	if built[0].activations.Load() != 1 {
		t.Fatal("startup did not activate its part")
	}

	changed := Update{Config: parseTestConfig(t, strings.Replace(testDocument, "192.0.2.2:8000", "192.0.2.3:8000", 1))}
	runtime.warm = func(context.Context, *Candidate) error { return errors.New("candidate rejected") }
	if _, applyErr := manager.Apply(t.Context(), changed); applyErr == nil {
		t.Fatal("expected warm rejection")
	}
	if built[1].activations.Load() != 0 || built[1].closed.Load() != 1 {
		t.Fatal("a rejected candidate published effects or kept its part open")
	}
	if manager.Active() != first || built[0].activations.Load() != 1 {
		t.Fatal("rejection changed active side effects")
	}

	runtime.warm = nil
	second, err := manager.Apply(t.Context(), changed)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = second.Release(context.Background()) }()
	if built[2].activations.Load() != 1 {
		t.Fatal("successful reload did not activate its part")
	}
	recipeOnly := Update{Config: parseTestConfig(t, strings.Replace(strings.Replace(testDocument,
		"192.0.2.2:8000", "192.0.2.3:8000", 1), "name: coding-decision", "name: coding-v2", 1))}
	third, err := manager.Apply(t.Context(), recipeOnly)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = third.Release(context.Background()) }()
	if third.Part(ComponentBackendTelemetry) != second.Part(ComponentBackendTelemetry) || built[2].activations.Load() != 2 {
		t.Fatal("reusing an unchanged part did not preserve and reactivate its effects")
	}
}

func TestTelemetrySettingsRebindUpstreamButUnrelatedSettingsDoNot(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelConfig = map[string]config.ModelParams{"m": {}}
	cfg.VLLMEndpoints = []config.VLLMEndpoint{{
		Name: "m_a", Model: "m", Address: "127.0.0.1", Port: 8000,
		Telemetry: config.EndpointTelemetry{Engine: "vllm"},
	}}
	cfg.Observability.BackendTelemetry = config.BackendTelemetryConfig{Enabled: true, PollInterval: "2s"}
	snapshot := func(cfg *config.RouterConfig) *Snapshot {
		resources, err := Compile(cfg)
		if err != nil {
			t.Fatal(err)
		}
		return &Snapshot{config: cfg, resources: resources}
	}
	base := snapshot(cfg)
	next := *cfg
	next.Observability.BackendTelemetry.PollInterval = "3s"
	settings := snapshot(&next)
	for _, component := range []Component{ComponentBackendTelemetry, ComponentUpstream} {
		if base.ComponentKey(component) == settings.ComponentKey(component) {
			t.Fatalf("%s retained an old telemetry generation after a settings change", component)
		}
	}
	next = *cfg
	next.Observability.BackendTelemetry.Enabled = false
	disabled := snapshot(&next)
	if base.ComponentKey(ComponentUpstream) == disabled.ComponentKey(ComponentUpstream) {
		t.Fatal("disabling telemetry did not rebind upstream")
	}
	next = *cfg
	next.ListBackendModels = true
	unrelated := snapshot(&next)
	for _, component := range []Component{ComponentBackendTelemetry, ComponentUpstream} {
		if base.ComponentKey(component) != unrelated.ComponentKey(component) {
			t.Fatalf("an unrelated setting rebuilt %s", component)
		}
	}
}
