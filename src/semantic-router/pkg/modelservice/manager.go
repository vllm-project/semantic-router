package modelservice

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
	startingPollInterval  = 500 * time.Millisecond
	readyPollInterval     = 5 * time.Second
	healthTimeout         = 2 * time.Second
)

// DeploymentStatus is the observable state of one deployment.
type DeploymentStatus struct {
	Name     string `json:"name"`
	Managed  bool   `json:"managed"`
	Endpoint string `json:"endpoint"`
	Ready    bool   `json:"ready"`
	State    string `json:"state"`
}

type deployment struct {
	name       string
	spec       config.ModelDeployment
	client     *Client
	managed    bool
	ready      atomic.Bool
	state      atomic.Value
	cancel     context.CancelFunc
	done       chan struct{}
	supervisor *supervisor
}

func (d *deployment) status() DeploymentStatus {
	state, _ := d.state.Load().(string)
	return DeploymentStatus{Name: d.name, Managed: d.managed, Endpoint: d.client.Endpoint(), Ready: d.ready.Load(), State: state}
}

// Manager owns the model_runtime deployments of the running configuration.
type Manager struct {
	mu          sync.RWMutex
	deployments map[string]*deployment
	runtimeDir  string
	command     []string
	cacheDir    string
	closed      bool
}

// NewManager prepares a manager; Reconcile starts deployments.
func NewManager() *Manager {
	command := strings.Fields(os.Getenv(RuntimeCommandEnv))
	if len(command) == 0 {
		command = []string{defaultRuntimeCommand}
	}
	runtimeDir := os.Getenv(RuntimeDirEnv)
	if runtimeDir == "" {
		runtimeDir = filepath.Join(os.TempDir(), fmt.Sprintf("vllm-sr-runtime-%d", os.Getpid()))
	}
	return &Manager{
		deployments: make(map[string]*deployment),
		runtimeDir:  runtimeDir,
		command:     command,
		cacheDir:    os.Getenv(RuntimeCacheEnv),
	}
}

// Reconcile starts deployments the configuration uses, restarts the ones whose
// settings changed, and stops the ones it no longer uses.
func (m *Manager) Reconcile(cfg *config.RouterConfig) error {
	desired := config.ModelRuntimeDeploymentsInUse(cfg)
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return fmt.Errorf("model runtime manager is shut down")
	}
	for name, current := range m.deployments {
		spec, keep := desired[name]
		if keep && reflect.DeepEqual(spec, current.spec) {
			continue
		}
		current.stop()
		delete(m.deployments, name)
	}
	var errs []string
	for _, name := range sortedNames(desired) {
		if _, running := m.deployments[name]; running {
			continue
		}
		started, err := m.start(name, desired[name])
		if err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", name, err))
			continue
		}
		m.deployments[name] = started
	}
	if len(errs) > 0 {
		return fmt.Errorf("model runtime deployments: %s", strings.Join(errs, "; "))
	}
	return nil
}

func (m *Manager) start(name string, spec config.ModelDeployment) (*deployment, error) {
	endpoint := strings.TrimSpace(spec.Endpoint)
	managed := endpoint == ""
	var socket string
	if managed {
		if err := os.MkdirAll(m.runtimeDir, 0o700); err != nil {
			return nil, err
		}
		socket = filepath.Join(m.runtimeDir, name+".sock")
		endpoint = "unix://" + socket
	}
	client, err := NewClient(endpoint)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	d := &deployment{name: name, spec: spec, client: client, managed: managed, cancel: cancel, done: make(chan struct{})}
	d.state.Store("starting")
	readyGauge.WithLabelValues(name).Set(0)
	var wg sync.WaitGroup
	if managed {
		d.supervisor = &supervisor{name: name, command: managedCommand(m.command, spec, socket, m.cacheDir), env: os.Environ(), socket: socket}
		wg.Add(1)
		go func() {
			defer wg.Done()
			d.supervisor.run(ctx)
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		d.watch(ctx)
	}()
	go func() {
		wg.Wait()
		close(d.done)
	}()
	logging.ComponentEvent("model_runtime", "deployment_started", map[string]interface{}{
		"deployment": name, "managed": managed, "endpoint": endpoint, "artifact": spec.Artifact, "profile": spec.Profile,
	})
	return d, nil
}

// watch polls readiness: fast while starting, slower once ready.
func (d *deployment) watch(ctx context.Context) {
	for {
		probe, cancel := context.WithTimeout(ctx, healthTimeout)
		ready, state, err := d.client.Ready(probe)
		cancel()
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			state = "unreachable"
		}
		if ready != d.ready.Load() {
			logging.ComponentEvent("model_runtime", "deployment_readiness_changed", map[string]interface{}{
				"deployment": d.name, "ready": ready, "state": state,
			})
		}
		d.ready.Store(ready)
		d.state.Store(state)
		if ready {
			readyGauge.WithLabelValues(d.name).Set(1)
		} else {
			readyGauge.WithLabelValues(d.name).Set(0)
		}
		interval := startingPollInterval
		if ready {
			interval = readyPollInterval
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(interval):
		}
	}
}

func (d *deployment) stop() {
	d.cancel()
	<-d.done
	d.ready.Store(false)
	readyGauge.WithLabelValues(d.name).Set(0)
	logging.ComponentEvent("model_runtime", "deployment_stopped", map[string]interface{}{"deployment": d.name})
}

// Decide answers the request through a deployment. A deployment that is not
// ready fails at once with ErrUnavailable so callers fail open without waiting.
func (m *Manager) Decide(ctx context.Context, name string, request Request) (Response, error) {
	m.mu.RLock()
	d := m.deployments[name]
	m.mu.RUnlock()
	if d == nil {
		requestsTotal.WithLabelValues(name, ErrorReason(ErrUnknownDeployment)).Inc()
		return Response{}, ErrUnknownDeployment
	}
	if !d.ready.Load() {
		requestsTotal.WithLabelValues(name, ErrorReason(ErrUnavailable)).Inc()
		return Response{}, ErrUnavailable
	}
	started := time.Now()
	response, err := d.client.Decide(ctx, request)
	requestDuration.WithLabelValues(name).Observe(time.Since(started).Seconds())
	requestsTotal.WithLabelValues(name, ErrorReason(err)).Inc()
	return response, err
}

// Statuses reports every deployment, sorted by name.
func (m *Manager) Statuses() []DeploymentStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()
	statuses := make([]DeploymentStatus, 0, len(m.deployments))
	for _, name := range sortedNames(m.deployments) {
		statuses = append(statuses, m.deployments[name].status())
	}
	return statuses
}

// Shutdown stops every deployment and its managed process.
func (m *Manager) Shutdown(ctx context.Context) error {
	m.mu.Lock()
	m.closed = true
	deployments := m.deployments
	m.deployments = map[string]*deployment{}
	m.mu.Unlock()
	done := make(chan struct{})
	go func() {
		for _, d := range deployments {
			d.stop()
		}
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func sortedNames[T any](values map[string]T) []string {
	names := make([]string, 0, len(values))
	for name := range values {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

var defaultManager atomic.Pointer[Manager]

// SetDefault installs the process-wide manager used by signals and selectors.
func SetDefault(manager *Manager) { defaultManager.Store(manager) }

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
