package modelservice

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	// RuntimeCommandEnv overrides the managed runtime command (space-separated),
	// for example "python3 -m vllm_sr_runtime".
	RuntimeCommandEnv = "VLLM_SR_RUNTIME_COMMAND"
	// RuntimeDirEnv overrides the private directory that holds runtime sockets.
	RuntimeDirEnv = "VLLM_SR_RUNTIME_DIR"
	// RuntimeCacheEnv sets the Hugging Face cache directory for managed runtimes.
	RuntimeCacheEnv = "VLLM_SR_RUNTIME_CACHE_DIR"

	defaultRuntimeCommand = "vllm-sr-runtime"
)

// Manager owns the runtime processes of every router generation. Each
// generation holds a Lease; processes are shared by composition and stop
// when the last lease that uses them closes.
type Manager struct {
	mu         sync.Mutex
	groups     map[string]*group
	published  *Lease
	runtimeDir string
	command    []string
	cacheDir   string
	closed     bool
}

// NewManager prepares a manager; Acquire and Reconcile start processes.
func NewManager() *Manager {
	command := strings.Fields(os.Getenv(RuntimeCommandEnv))
	if len(command) == 0 {
		command = []string{defaultRuntimeCommand}
	}
	runtimeDir := os.Getenv(RuntimeDirEnv)
	if runtimeDir == "" {
		runtimeDir = filepath.Join(os.TempDir(), fmt.Sprintf("vllm-sr-runtime-%d", os.Getpid()))
	}
	return &Manager{groups: make(map[string]*group), runtimeDir: runtimeDir, command: command, cacheDir: os.Getenv(RuntimeCacheEnv)}
}

// Acquire returns a lease on the model_runtime deployments the configuration
// uses, starting the processes no other generation runs yet. It does not wait
// for readiness: calls fail open until a deployment is ready, and Card waits.
func (m *Manager) Acquire(cfg *config.RouterConfig) (*Lease, error) {
	plans := planProcesses(config.ModelRuntimeDeploymentsInUse(cfg), m.command, m.cacheDir)
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, fmt.Errorf("model runtime manager is shut down")
	}
	lease := &Lease{manager: m, members: make(map[string]member)}
	for _, plan := range plans {
		g := m.groups[plan.key]
		if g == nil {
			started, err := m.startGroupLocked(plan)
			if err != nil {
				m.releaseLocked(lease.groups)
				return nil, fmt.Errorf("model runtime process %s: %w", plan.name, err)
			}
			g = started
			m.groups[plan.key] = g
		}
		g.refs++
		lease.groups = append(lease.groups, g)
		for deployment, model := range plan.members {
			lease.members[deployment] = member{group: g, served: g.models[model]}
		}
	}
	return lease, nil
}

func (m *Manager) startGroupLocked(plan *processPlan) (*group, error) {
	if plan.endpoint != "" {
		client, err := NewClient(plan.endpoint)
		if err != nil {
			return nil, err
		}
		g := newGroup(plan, client, false)
		g.start()
		return g, nil
	}
	if err := os.MkdirAll(m.runtimeDir, 0o700); err != nil {
		return nil, err
	}
	if err := os.Chmod(m.runtimeDir, 0o700); err != nil {
		return nil, err
	}
	socket, modelsFile := plan.files(m.runtimeDir)
	if err := plan.writeModelsFile(modelsFile); err != nil {
		return nil, err
	}
	client, err := NewClient("unix://" + socket)
	if err != nil {
		return nil, err
	}
	g := newGroup(plan, client, true)
	g.modelsFile = modelsFile
	g.supervisor = &supervisor{
		process: plan.name, deployments: plan.deployments(), socket: socket, env: os.Environ(),
		command: managedCommand(m.command, socket, modelsFile, m.cacheDir), onExit: g.processExited,
	}
	g.start()
	return g, nil
}

// release drops a lease's references and stops the processes nobody uses.
func (m *Manager) release(groups []*group) {
	m.mu.Lock()
	stopped := m.releaseLocked(groups)
	m.mu.Unlock()
	stopGroups(stopped)
}

func (m *Manager) releaseLocked(groups []*group) []*group {
	var stopped []*group
	for _, g := range groups {
		g.refs--
		if g.refs == 0 {
			delete(m.groups, g.plan.key)
			stopped = append(stopped, g)
		}
	}
	return stopped
}

func stopGroups(groups []*group) {
	var wg sync.WaitGroup
	for _, g := range groups {
		wg.Add(1)
		go func(g *group) {
			defer wg.Done()
			g.stop()
		}(g)
	}
	wg.Wait()
}

// Reconcile keeps a lease on the published configuration's deployments for
// the process-wide Default and releases the lease of the configuration it replaces.
func (m *Manager) Reconcile(cfg *config.RouterConfig) error {
	lease, err := m.Acquire(cfg)
	if err != nil {
		return err
	}
	m.mu.Lock()
	previous := m.published
	m.published = lease
	m.mu.Unlock()
	return previous.Close()
}

// Published returns the lease Reconcile keeps (nil before the first Reconcile).
func (m *Manager) Published() *Lease {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.published
}

// Statuses reports the published configuration's deployments, sorted by name.
func (m *Manager) Statuses() []DeploymentStatus {
	return m.Published().Statuses()
}

// Decide answers through the published configuration's deployments.
func (m *Manager) Decide(ctx context.Context, deployment string, request Request) (Response, error) {
	lease := m.Published()
	if lease == nil {
		return Response{}, ErrUnavailable
	}
	return lease.Decide(ctx, deployment, request)
}

// Shutdown stops every process, whatever leases remain.
func (m *Manager) Shutdown(ctx context.Context) error {
	m.mu.Lock()
	m.closed = true
	groups := make([]*group, 0, len(m.groups))
	for _, g := range m.groups {
		// Leases closed after shutdown must not stop a group again.
		g.refs = 0
		groups = append(groups, g)
	}
	m.groups = map[string]*group{}
	m.published = nil
	m.mu.Unlock()
	done := make(chan struct{})
	go func() {
		stopGroups(groups)
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return errors.Join(ctx.Err(), errors.New("model runtime processes are still stopping"))
	}
}

var defaultManager atomic.Pointer[Manager]

// SetDefault installs the process-wide manager.
func SetDefault(manager *Manager) { defaultManager.Store(manager) }

// DefaultManager returns the process-wide manager, or nil before one is installed.
func DefaultManager() *Manager { return defaultManager.Load() }

// Default returns the process-wide decider. Before a manager is installed it
// answers ErrUnavailable, so request paths fail open.
func Default() Decider {
	if manager := defaultManager.Load(); manager != nil {
		return manager
	}
	return unavailable{}
}

type unavailable struct{}

func (unavailable) Decide(context.Context, string, Request) (Response, error) {
	return Response{}, ErrUnavailable
}
