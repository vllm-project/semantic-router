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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	// RuntimeCommandEnv overrides the managed runtime command (space-separated),
	// for example "python3 -m vllm_srun".
	RuntimeCommandEnv = "VLLM_SRUN_COMMAND"
	// RuntimeDirEnv overrides the private directory that holds runtime sockets.
	RuntimeDirEnv = "VLLM_SRUN_DIR"
	// RuntimeCacheEnv sets the Hugging Face cache directory for managed runtimes.
	RuntimeCacheEnv = "VLLM_SRUN_CACHE_DIR"

	defaultRuntimeCommand = "vllm-srun"
)

// Manager owns the runtime processes of every router generation. Each
// generation holds a Lease; unchanged logical workers keep their identities
// across modes and generations, and stop after their last reference drains.
type Manager struct {
	mu              sync.Mutex
	groups          map[string]*group
	attachedClients map[string]*Client
	published       *Lease
	runtimeDir      string
	command         []string
	cacheDir        string
	cores           int
	closed          bool

	dispatchSequence uint64

	autoOnce sync.Once
	auto     string
}

// NewManager prepares a manager; Acquire and Reconcile start processes.
func NewManager() *Manager {
	command := strings.Fields(os.Getenv(RuntimeCommandEnv))
	if len(command) == 0 {
		command = []string{defaultRuntimeCommand}
	}
	runtimeDir := os.Getenv(RuntimeDirEnv)
	if runtimeDir == "" {
		runtimeDir = filepath.Join(os.TempDir(), fmt.Sprintf("vllm-srun-%d", os.Getpid()))
	}
	return &Manager{groups: make(map[string]*group), runtimeDir: runtimeDir, command: command, cacheDir: os.Getenv(RuntimeCacheEnv), cores: cpuCores()}
}

// Acquire returns a lease on the model_runtime deployments the configuration
// uses, starting the processes no other generation runs yet. It does not wait
// for readiness: calls fail open until a deployment is ready, and Card waits.
func (m *Manager) Acquire(cfg *config.RouterConfig) (*Lease, error) {
	return m.AcquireDeployments(config.ModelRuntimeDeploymentsInUse(cfg))
}

// AcquireDeployments returns a lease on an explicit set of deployments.
func (m *Manager) AcquireDeployments(deployments map[string]config.ModelDeployment) (*Lease, error) {
	auto := m.resolveAuto(deployments)
	if err := refuseGPUOnlyOnCPU(deployments, auto); err != nil {
		return nil, err
	}
	plans := planProcesses(deployments, m.command, m.cacheDir, m.cores, auto)
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
				stopped := m.releaseLocked(lease.groups)
				go stopGroups(stopped)
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
	lease.assemblePools(deployments)
	return lease, nil
}

// extend adds one deployment to a lease in a process of its own.
func (m *Manager) extend(lease *Lease, name string, deployment config.ModelDeployment) error {
	added, err := m.AcquireDeployments(map[string]config.ModelDeployment{name: deployment})
	if err != nil {
		return err
	}
	lease.mu.Lock()
	if _, ok := lease.members[name]; ok {
		lease.mu.Unlock()
		return added.Close()
	}
	if lease.closed {
		lease.mu.Unlock()
		_ = added.Close()
		return ErrUnavailable
	}
	lease.groups = append(lease.groups, added.groups...)
	lease.members[name] = added.members[name]
	added.groups = nil
	lease.mu.Unlock()
	logging.ComponentWarnEvent("model_runtime", "deployment_outside_generation_plan", map[string]interface{}{"deployment": name})
	return nil
}

// resolveAuto returns the device that managed deployments on auto are planned
// on: the runtime's answer, asked the first time one is planned and kept for
// the manager's lifetime, so plans stay stable. It is empty when the runtime
// could not answer; each worker then resolves auto during its own startup.
func (m *Manager) resolveAuto(deployments map[string]config.ModelDeployment) string {
	for _, deployment := range deployments {
		needsAuto := false
		for _, placement := range deployment.Placements() {
			if placement.Endpoint == "" && placement.Device == autoDevice {
				needsAuto = true
			}
		}
		if !deployment.IsModelRuntime() || !needsAuto {
			continue
		}
		m.autoOnce.Do(func() {
			device, err := queryAutoDevice(m.command)
			if err != nil {
				logging.ComponentWarnEvent("model_runtime", "auto_device_unresolved", map[string]interface{}{
					"error": err.Error(), "fallback": "each auto worker resolves placement at startup",
				})
				return
			}
			m.auto = device
			logging.ComponentEvent("model_runtime", "auto_device_resolved", map[string]interface{}{"device": device})
		})
		return m.auto
	}
	return ""
}

func (m *Manager) startGroupLocked(plan *processPlan) (*group, error) {
	if plan.endpoint != "" {
		if m.attachedClients == nil {
			m.attachedClients = make(map[string]*Client)
		}
		client := m.attachedClients[plan.endpoint]
		if client == nil {
			var err error
			client, err = NewClient(plan.endpoint)
			if err != nil {
				return nil, err
			}
			m.attachedClients[plan.endpoint] = client
		}
		g := newGroup(plan, client, false)
		g.start()
		return g, nil
	}
	if err := privateDir(m.runtimeDir); err != nil {
		return nil, err
	}
	socket, modelsFile := plan.files(m.runtimeDir)
	if len(socket) > maxSocketPath {
		dir := shortSocketDir(m.runtimeDir)
		if err := privateDir(dir); err != nil {
			return nil, err
		}
		socket, _ = plan.files(dir)
	}
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
		process: plan.name, deployments: []string{plan.logical}, socket: socket, env: os.Environ(),
		command: managedCommand(m.command, socket, modelsFile, m.cacheDir, plan.threads), onExit: g.processExited,
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
