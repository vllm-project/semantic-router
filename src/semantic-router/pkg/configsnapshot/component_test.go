package configsnapshot

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func compiledSnapshot(t *testing.T, document string) *Snapshot {
	t.Helper()
	cfg := parseTestConfig(t, document)
	m := NewManager(Options{})
	snapshot, err := m.Install(context.Background(), Update{Origin: Origin{Source: SourceStartup}, Config: cfg})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

// The dependency graph decides what a change rebuilds: each case lists the
// components whose inputs move.
func TestComponentKeysFollowTheDependencyGraph(t *testing.T) {
	t.Setenv("TEST_HOSTED_KEY", "hosted-secret-value")
	base := compiledSnapshot(t, testDocument)
	for _, tc := range []struct {
		name    string
		edit    func(string) string
		changed []Component
	}{
		{
			name:    "endpoint only",
			edit:    func(doc string) string { return strings.Replace(doc, "192.0.2.2:8000", "192.0.2.3:8000", 1) },
			changed: []Component{ComponentRouter, ComponentUpstream},
		},
		{
			name: "recipe only",
			edit: func(doc string) string {
				return strings.Replace(doc, "name: coding-decision", "name: coding-decision-2", 1)
			},
			changed: []Component{ComponentRouter, ComponentSignals},
		},
		{
			name: "global setting only",
			edit: func(doc string) string {
				return strings.Replace(doc, "recipes:", "global:\n  router:\n    list_backend_models: true\nrecipes:", 1)
			},
			changed: []Component{ComponentRouter, ComponentSignals},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			next := compiledSnapshot(t, tc.edit(testDocument))
			var changed []Component
			for _, component := range []Component{ComponentRouter, ComponentSignals, ComponentUpstream} {
				if base.ComponentKey(component) != next.ComponentKey(component) {
					changed = append(changed, component)
				}
			}
			if !reflect.DeepEqual(changed, tc.changed) {
				t.Fatalf("changed components = %v, want %v", changed, tc.changed)
			}
		})
	}
}

func TestClusterOrderMovesTheUpstreamKey(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelConfig = map[string]config.ModelParams{"a": {}, "b": {}}
	cfg.VLLMEndpoints = []config.VLLMEndpoint{
		{Name: "e", Model: "a", Address: "a.example", Port: 80},
		{Name: "e", Model: "b", Address: "b.example", Port: 80},
	}
	cfg.ProviderModelOrder = []string{"a", "b"}
	first, err := Compile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	reordered := *cfg
	reordered.ProviderModelOrder = []string{"b", "a"}
	second, err := Compile(&reordered)
	if err != nil {
		t.Fatal(err)
	}
	a := &Snapshot{resources: first}
	b := &Snapshot{resources: second}
	if a.ComponentKey(ComponentUpstream) == b.ComponentKey(ComponentUpstream) {
		t.Fatal("moving the default route's model kept the upstream key")
	}
}

func TestOwnedFieldsNameRealConfigurationFields(t *testing.T) {
	fields := make(map[string]bool)
	var walk func(reflect.Type)
	walk = func(typ reflect.Type) {
		for i := range typ.NumField() {
			field := typ.Field(i)
			fields[typ.Name()+"."+field.Name] = true
			if field.Anonymous && field.Type.Kind() == reflect.Struct {
				walk(field.Type)
			}
		}
	}
	walk(reflect.TypeOf(config.RouterConfig{}))
	for owned := range ownedFields {
		if !fields[owned] {
			t.Errorf("ownedFields names %q, which RouterConfig does not have", owned)
		}
	}
}

// countingPart is a part that records how often it was closed.
type countingPart struct {
	id     int
	closed atomic.Int32
}

func (p *countingPart) Close(context.Context) error {
	p.closed.Add(1)
	return nil
}

type partRecorder struct {
	builds    atomic.Int32
	previous  []Part
	lastBuilt *countingPart
	validate  func(candidate, active *Snapshot) error
	warm      func(Part) error
}

func (r *partRecorder) builder() PartBuilder {
	return PartBuilder{
		Component: ComponentUpstream,
		Validate: func(candidate, active *Snapshot) error {
			if r.validate != nil {
				return r.validate(candidate, active)
			}
			return nil
		},
		Build: func(_ context.Context, _ *Snapshot, previous Part) (Part, error) {
			r.previous = append(r.previous, previous)
			r.lastBuilt = &countingPart{id: int(r.builds.Add(1))}
			return r.lastBuilt, nil
		},
		Warm: func(_ context.Context, part Part) error {
			if r.warm != nil {
				return r.warm(part)
			}
			return nil
		},
	}
}

func partsManager(t *testing.T, recorder *partRecorder, runtime *fakeRuntime) *Manager {
	t.Helper()
	m := NewManager(Options{Runtime: runtime, Parts: []PartBuilder{recorder.builder()}})
	t.Setenv("TEST_HOSTED_KEY", "hosted-secret-value")
	if _, err := m.Install(context.Background(), Update{
		Origin: Origin{Source: SourceStartup}, Config: parseTestConfig(t, testDocument),
	}); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestPartsAreKeptWhileTheirResourcesAreUnchanged(t *testing.T) {
	recorder := &partRecorder{}
	m := partsManager(t, recorder, &fakeRuntime{})
	first := m.Active()
	firstPart := first.Part(ComponentUpstream).(*countingPart)

	recipeOnly := parseTestConfig(t, strings.Replace(testDocument, "name: coding-decision", "name: coding-v2", 1))
	second, err := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceFile}, Config: recipeOnly})
	if err != nil {
		t.Fatal(err)
	}
	if second.Part(ComponentUpstream) != firstPart || recorder.builds.Load() != 1 {
		t.Fatalf("a recipe-only change rebuilt the upstream part (%d builds)", recorder.builds.Load())
	}
	if reused := second.Reused(); len(reused) != 1 || reused[0] != ComponentUpstream {
		t.Fatalf("Reused() = %v", reused)
	}
	if latest := m.Status().Latest; len(latest.Reused) != 1 || latest.Reused[0] != ComponentUpstream {
		t.Fatalf("attempt reuse = %v", latest.Reused)
	}

	endpointChange := parseTestConfig(t, strings.Replace(testDocument, "192.0.2.2:8000", "192.0.2.3:8000", 1))
	third, err := m.Apply(context.Background(), Update{Origin: Origin{Source: SourceFile}, Config: endpointChange})
	if err != nil {
		t.Fatal(err)
	}
	thirdPart := third.Part(ComponentUpstream).(*countingPart)
	if thirdPart == firstPart || recorder.previous[1] != Part(firstPart) {
		t.Fatal("an endpoint change did not build a new part from the previous one")
	}

	// The first part closes only after both snapshots that served with it
	// have drained.
	if err := first.Release(context.Background()); err != nil || firstPart.closed.Load() != 0 {
		t.Fatalf("part closed while a snapshot still serves with it: %v", err)
	}
	if err := second.Release(context.Background()); err != nil || firstPart.closed.Load() != 1 {
		t.Fatalf("part close count = %d after its last snapshot drained", firstPart.closed.Load())
	}
	_ = second.Release(context.Background())
	if firstPart.closed.Load() != 1 || thirdPart.closed.Load() != 0 {
		t.Fatal("Release was not idempotent, or closed the active part")
	}
}

func TestPartFailuresRejectTheUpdateAndCloseWhatWasBuilt(t *testing.T) {
	restart := Reject(StageValidate, CodeRestartRequired, errors.New("listener public moved to port 9000"))
	recorder := &partRecorder{}
	runtime := &fakeRuntime{}
	m := partsManager(t, recorder, runtime)
	active := m.Active()
	endpointChange := func() Update {
		return Update{Origin: Origin{Source: SourceFile}, Config: parseTestConfig(t, strings.Replace(testDocument, "192.0.2.2:8000", "192.0.2.4:8000", 1))}
	}

	recorder.validate = func(_, _ *Snapshot) error { return restart }
	_, err := m.Apply(context.Background(), endpointChange())
	if reasons := ReasonsOf(err); len(reasons) != 1 || reasons[0].Code != CodeRestartRequired || recorder.builds.Load() != 1 {
		t.Fatalf("validate rejection = %v (builds %d)", err, recorder.builds.Load())
	}

	recorder.validate = nil
	var warmed *countingPart
	recorder.warm = func(part Part) error {
		warmed = part.(*countingPart)
		return errors.New("health checks did not finish")
	}
	_, err = m.Apply(context.Background(), endpointChange())
	if reasons := ReasonsOf(err); len(reasons) != 1 || reasons[0].Stage != StageWarm || warmed == nil || warmed.closed.Load() != 1 {
		t.Fatalf("warm rejection = %v; built part closed %d times", err, warmed.closed.Load())
	}
	if runtime.validated.Load() != 2 {
		t.Fatalf("the router runtime validated %d candidates, want 2", runtime.validated.Load())
	}

	recorder.warm = nil
	runtime.warm = func(context.Context, *Candidate) error { return errors.New("classifier unavailable") }
	_, err = m.Apply(context.Background(), endpointChange())
	activePart := m.Active().Part(ComponentUpstream).(*countingPart)
	if err == nil || m.Active() != active || activePart.closed.Load() != 0 {
		t.Fatalf("router failure: err %v, active changed %v", err, m.Active() != active)
	}
	if recorder.lastBuilt == activePart || recorder.lastBuilt.closed.Load() != 1 {
		t.Fatalf("the rejected candidate's part closed %d times, want 1", recorder.lastBuilt.closed.Load())
	}
	if last := recorder.previous[len(recorder.previous)-1]; last != Part(activePart) {
		t.Fatal("the rejected candidate's part was not built from the active one")
	}
}
