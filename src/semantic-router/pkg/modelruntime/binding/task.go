package binding

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// Registry stores task definitions only. Type erasure is confined to metadata
// lookup; input, output, provider preparation and inference stay typed.
type Registry struct {
	mu       sync.RWMutex
	tasks    map[string]any
	observer Observer
}

func NewRegistry(observers ...Observer) *Registry {
	r := &Registry{tasks: make(map[string]any)}
	if len(observers) > 0 {
		r.observer = observers[0]
	}
	return r
}

// Event contains execution facts only, never prompts, credentials or results.
type Event struct {
	Identity      Identity
	Capability    Capability
	State         string
	Duration      time.Duration
	AdmissionWait time.Duration
	Executed      bool
	Input         *tasks.InputUsage
	Error         error
}
type Observer func(Event)

type Task[Input, Output any] struct {
	mu             sync.RWMutex
	contract       string
	validateInput  func(Input) error
	validateOutput func(Input, Output) error
	providers      map[string]func(context.Context, Identity) (Prepared[Input, Output], error)
	observer       Observer
}

// Prepared is the typed result of loading and warming a task adapter. The
// resource is owned by the caller until Bind transfers it to a Resolved handle.
type Prepared[Input, Output any] struct {
	Capability Capability
	Resource   *Resource
	Infer      func(context.Context, io.Closer, Input) (Output, error)
}

func (t *Task[Input, Output]) RegisterProvider(name string, prepare func(context.Context, Identity) (Prepared[Input, Output], error)) error {
	if name == "" || prepare == nil {
		return fmt.Errorf("provider name and preparation function are required")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.providers == nil {
		t.providers = make(map[string]func(context.Context, Identity) (Prepared[Input, Output], error))
	}
	if _, exists := t.providers[name]; exists {
		return fmt.Errorf("provider %q is already registered for %q", name, t.contract)
	}
	t.providers[name] = prepare
	return nil
}

// Bind resolves provider registration and prepares the typed handle once,
// before generation publication. Requests only call the returned handle.
func (t *Task[Input, Output]) Bind(ctx context.Context, provider string, id Identity) (*Resolved[Input, Output], error) {
	t.mu.RLock()
	prepare, exists := t.providers[provider]
	t.mu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("%w: provider %q is not registered for %q", ErrCapability, provider, t.contract)
	}
	if id.Contract != t.contract {
		return nil, fmt.Errorf("%w: unexpected task contract", ErrCapability)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	prepared, err := prepare(ctx, id)
	if err == nil {
		err = ctx.Err()
	}
	if err == nil && prepared.Capability.Provider != provider {
		err = fmt.Errorf("%w: effective provider differs from requested provider", ErrCapability)
	}
	if err != nil {
		if prepared.Resource != nil {
			_ = prepared.Resource.Close()
		}
		return nil, err
	}
	resolved, err := t.Resolve(id, prepared.Capability, prepared.Resource, prepared.Infer)
	if err != nil && prepared.Resource != nil {
		_ = prepared.Resource.Close()
	}
	return resolved, err
}

func Register[Input, Output any](r *Registry, contract string, input func(Input) error, output func(Input, Output) error) (*Task[Input, Output], error) {
	return RegisterTask(r, contract, contract, input, output)
}

// RegisterTask allows different typed inputs to share a result contract. A
// grounded text task and a plain text task can both return token_spans.v1;
// their registry identities and input validators remain distinct.
func RegisterTask[Input, Output any](r *Registry, taskID, contract string, input func(Input) error, output func(Input, Output) error) (*Task[Input, Output], error) {
	if taskID == "" || contract == "" || input == nil || output == nil {
		return nil, fmt.Errorf("task contract and validators are required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tasks[taskID]; exists {
		return nil, fmt.Errorf("task %q is already registered", taskID)
	}
	task := &Task[Input, Output]{contract: contract, validateInput: input, validateOutput: output, observer: r.observer}
	r.tasks[taskID] = task
	return task, nil
}

func Lookup[Input, Output any](r *Registry, contract string) (*Task[Input, Output], error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	task, ok := r.tasks[contract].(*Task[Input, Output])
	if !ok {
		return nil, fmt.Errorf("%w: task %q is absent or has different input/output types", ErrCapability, contract)
	}
	return task, nil
}

// Resolve validates one provider-prepared handle before it is published. The
// provider owns resource cleanup if resolution fails; a successful binding
// transfers that reference to its generation's resource scope.
func (t *Task[Input, Output]) Resolve(id Identity, capability Capability, resource *Resource, infer func(context.Context, io.Closer, Input) (Output, error)) (*Resolved[Input, Output], error) {
	if id.Contract != t.contract {
		return nil, fmt.Errorf("%w: unexpected task contract", ErrCapability)
	}
	if err := capability.Validate(id); err != nil {
		return nil, err
	}
	if resource == nil || infer == nil {
		return nil, fmt.Errorf("prepared task resource and inference function are required")
	}
	capability = cloneCapability(capability)
	bound := &Resolved[Input, Output]{identity: id, capability: capability, resource: resource, task: t, infer: infer}
	bound.observe(Event{State: "resolved"})
	return bound, nil
}

// Resolved is immutable prepared state. A caller supplies its recipe on
// lookup; it cannot use physical sharing to access another recipe's binding.
type Resolved[Input, Output any] struct {
	identity   Identity
	capability Capability
	resource   *Resource
	task       *Task[Input, Output]
	infer      func(context.Context, io.Closer, Input) (Output, error)
	closeOnce  sync.Once
	closeErr   error
}

func (b *Resolved[Input, Output]) Identity() Identity { return b.identity }
func (b *Resolved[Input, Output]) Capability() Capability {
	return cloneCapability(b.capability)
}

func (b *Resolved[Input, Output]) Close() error {
	b.closeOnce.Do(func() {
		b.closeErr = b.resource.Close()
		b.observe(Event{State: "closed", Error: b.closeErr})
	})
	return b.closeErr
}

func (b *Resolved[Input, Output]) observe(event Event) {
	if b.task.observer == nil {
		return
	}
	event.Identity = b.identity
	event.Capability = b.Capability()
	b.task.observer(event)
}

// Ready is called after provider warmup succeeds, before generation publication.
func (b *Resolved[Input, Output]) Ready() { b.observe(Event{State: "ready"}) }

func (b *Resolved[Input, Output]) Call(ctx context.Context, recipe string, input Input) (output Output, callErr error) {
	start := time.Now()
	event := Event{State: "call"}
	defer func() {
		event.Duration = time.Since(start)
		event.Error = callErr
		if usage, ok := any(output).(interface{ InputMetadata() *tasks.InputUsage }); ok {
			event.Input = usage.InputMetadata()
		}
		b.observe(event)
	}()
	if recipe != b.identity.Recipe {
		return output, fmt.Errorf("%w: task binding does not belong to recipe %q", ErrCapability, recipe)
	}
	if err := b.task.validateInput(input); err != nil {
		return output, fmt.Errorf("%w: %w", ErrInvalidInput, err)
	}
	err := b.resource.Use(ctx, func(resource io.Closer) error {
		event.AdmissionWait = time.Since(start)
		event.Executed = true
		var err error
		output, err = b.infer(ctx, resource, input)
		if err != nil && !errors.Is(err, tasks.ErrTokenSpansTruncated) {
			return err
		}
		if validationErr := b.task.validateOutput(input, output); validationErr != nil {
			return fmt.Errorf("%w: %w", ErrInvalidResult, validationErr)
		}
		return err
	})
	return output, err
}
