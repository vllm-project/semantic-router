package modelservice

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	startingPollInterval = 500 * time.Millisecond
	readyPollInterval    = 5 * time.Second
	healthTimeout        = 2 * time.Second
	// A managed process that exits this many times, each within quickExitWindow
	// and before it was ever ready, fails the preparations waiting on it.
	maxQuickExits   = 3
	quickExitWindow = 30 * time.Second
	// A managed process whose every model failed to load this many times in a
	// row fails the preparations waiting on it; the supervisor keeps retrying.
	maxFailedLoads = 3
)

// group is one runtime process: managed (supervised on a private socket) or
// attached (an endpoint the Router does not manage). Router generations with
// the same composition share a group; the Manager counts their references.
type group struct {
	plan       *processPlan
	client     *Client
	managed    bool
	supervisor *supervisor
	modelsFile string
	cancel     context.CancelFunc
	done       chan struct{}
	probe      chan struct{}
	refs       int // guarded by Manager.mu

	mu          sync.Mutex
	changed     chan struct{}
	models      map[string]*servedModel
	failure     error
	quickExits  int
	failedLoads int
	restarts    int
	everReady   bool
}

// servedModel is one model of a process and the deployments that call it.
// ready is read lock-free on every call.
type servedModel struct {
	load        replicaLoad
	name        string
	deployments []string
	ready       atomic.Bool
	state       string
	reason      string
	card        *ModelCard
	cache       *resultCache
}

func newGroup(plan *processPlan, client *Client, managed bool) *group {
	g := &group{plan: plan, client: client, managed: managed, done: make(chan struct{}), probe: make(chan struct{}, 1), changed: make(chan struct{}), models: make(map[string]*servedModel)}
	for _, deployment := range plan.deployments() {
		model := plan.members[deployment]
		served := g.models[model]
		if served == nil {
			served = &servedModel{name: model, state: "starting", cache: newResultCache(ResultCacheEntries())}
			g.models[model] = served
		}
		served.deployments = append(served.deployments, plan.logical)
		g.recordReady(plan.logical, false)
	}
	return g
}

// start runs the watcher and, for a managed group, the supervised process.
func (g *group) start() {
	ctx, cancel := context.WithCancel(context.Background())
	g.cancel = cancel
	var wg sync.WaitGroup
	if g.supervisor != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			g.supervisor.run(ctx)
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		g.watch(ctx)
	}()
	go func() {
		wg.Wait()
		close(g.done)
	}()
	logging.ComponentEvent("model_runtime", "runtime_process_planned", map[string]interface{}{
		"process": g.plan.name, "managed": g.managed, "deployments": g.plan.deployments(), "models": len(g.models),
	})
}

// stop ends the watcher and the process (SIGTERM, then SIGKILL after the grace period).
func (g *group) stop() {
	g.cancel()
	<-g.done
	g.mu.Lock()
	for _, served := range g.models {
		served.ready.Store(false)
		g.recordReady(g.plan.logical, false)
		served.state = "stopped"
	}
	g.broadcastLocked()
	g.mu.Unlock()
	if g.modelsFile != "" {
		_ = os.Remove(g.modelsFile)
	}
	logging.ComponentEvent("model_runtime", "runtime_process_stopped", map[string]interface{}{"process": g.plan.name, "deployments": g.plan.deployments()})
}

// watch polls /health fast while a model is not ready and slowly once all
// are; a call that cannot reach the process asks for an early probe.
func (g *group) watch(ctx context.Context) {
	for {
		g.refresh(ctx)
		interval := readyPollInterval
		if !g.everyReady() {
			interval = startingPollInterval
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(interval):
		case <-g.probe:
		}
	}
}

// requestProbe wakes the watcher without blocking.
func (g *group) requestProbe() {
	select {
	case g.probe <- struct{}{}:
	default:
	}
}

func (g *group) everyReady() bool {
	for _, served := range g.models {
		if !served.ready.Load() {
			return false
		}
	}
	return true
}

func (g *group) refresh(ctx context.Context) {
	probe, cancel := context.WithTimeout(ctx, healthTimeout)
	defer cancel()
	health, err := g.client.health(probe)
	if ctx.Err() != nil {
		return
	}
	var cards map[string]ModelCard
	if err == nil && g.needsCards(health) {
		if list, cardErr := g.client.Models(probe); cardErr == nil {
			cards = make(map[string]ModelCard, len(list))
			for _, card := range list {
				cards[card.Id] = decodeCard(card)
			}
		}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	changed := false
	for _, served := range g.models {
		state, reason := "unreachable", ""
		if err == nil {
			state, reason = health.stateOf(served.name, len(g.models))
		}
		if card, ok := cards[served.name]; ok {
			served.card = &card
		}
		ready := state == "ready" && served.card != nil
		if ready != served.ready.Load() || state != served.state || reason != served.reason {
			changed = true
			if ready && !served.ready.Load() {
				served.cache.reset()
			}
			if ready != served.ready.Load() {
				logging.ComponentEvent("model_runtime", "deployment_readiness_changed", map[string]interface{}{
					"process": g.plan.name, "model": served.name, "deployments": served.deployments, "ready": ready, "state": state,
				})
			}
			if state == "incompatible" && served.state != state {
				logging.ComponentWarnEvent("model_runtime", "runtime_contract_incompatible", map[string]interface{}{
					"process": g.plan.name, "model": served.name, "deployments": served.deployments, "reason": reason,
				})
			}
			served.ready.Store(ready)
			served.state, served.reason = state, reason
			for _, deployment := range served.deployments {
				g.recordReady(deployment, ready)
			}
		}
		if ready {
			g.everReady = true
			g.failure = nil
			g.failedLoads = 0
		}
	}
	if changed {
		g.broadcastLocked()
	}
	g.recycleFailedLocked()
}

// recycleFailedLocked restarts a managed process in which every model failed
// to load, through the supervisor's back-off. A process that still serves any
// model keeps running.
func (g *group) recycleFailedLocked() {
	if !g.recyclesLocked() || !g.supervisor.recycle() {
		return
	}
	reasons := make(map[string]string, len(g.models))
	for _, served := range g.models {
		reasons[served.name] = served.reason
	}
	g.failedLoads++
	if g.failedLoads >= maxFailedLoads {
		g.failure = fmt.Errorf("every model failed to load %d times in a row: %v", g.failedLoads, reasons)
		g.broadcastLocked()
	}
	logging.ComponentWarnEvent("model_runtime", "runtime_process_recycled", map[string]interface{}{
		"process": g.plan.name, "deployments": g.plan.deployments(), "failed": reasons, "failed_loads": g.failedLoads,
	})
}

// recyclesLocked reports whether the group restarts its process for failed
// loads: it is managed and every model failed.
func (g *group) recyclesLocked() bool {
	if g.supervisor == nil || len(g.models) == 0 {
		return false
	}
	for _, served := range g.models {
		if served.state != "failed" {
			return false
		}
	}
	return true
}

// needsCards reports whether a model is ready without a card yet.
func (g *group) needsCards(health processHealth) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, served := range g.models {
		if state, _ := health.stateOf(served.name, len(g.models)); state == "ready" && (served.card == nil || !served.ready.Load()) {
			return true
		}
	}
	return false
}

// processExited records a managed process exit; repeated quick exits before
// any readiness, or a command that cannot run, fail the waiting preparations.
func (g *group) processExited(err error, ran time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.restarts++
	for _, served := range g.models {
		served.ready.Store(false)
		served.state = "restarting"
		for _, deployment := range served.deployments {
			g.recordReady(deployment, false)
		}
	}
	var execErr *exec.Error
	switch {
	case errors.As(err, &execErr):
		g.failure = fmt.Errorf("runtime command cannot run: %w", err)
	case errors.Is(err, errRecycled):
		// recycleFailedLocked counts failed loads.
	case !g.everReady && ran < quickExitWindow:
		g.quickExits++
		if g.quickExits >= maxQuickExits {
			g.failure = fmt.Errorf("runtime process exited %d times before it was ready: %w", g.quickExits, err)
		}
	}
	g.broadcastLocked()
}

func (g *group) broadcastLocked() {
	close(g.changed)
	g.changed = make(chan struct{})
}

// waitCard waits until a model is ready and returns its card. A failed model,
// a runtime of another contract major, a process that cannot start, or
// repeated failed loads end the wait at once;
// only while the group recycles a process whose every model failed does the
// wait continue.
func (g *group) waitCard(ctx context.Context, model string) (ModelCard, error) {
	for {
		g.mu.Lock()
		served := g.models[model]
		if served == nil {
			g.mu.Unlock()
			return ModelCard{}, ErrUnknownDeployment
		}
		if served.ready.Load() && served.card != nil {
			card := *served.card
			g.mu.Unlock()
			return card, nil
		}
		if served.state == "failed" && !g.recyclesLocked() {
			reason := served.reason
			g.mu.Unlock()
			return ModelCard{}, fmt.Errorf("%w: model %s failed to load: %s", ErrUnavailable, model, reason)
		}
		if served.state == "incompatible" {
			reason := served.reason
			g.mu.Unlock()
			return ModelCard{}, fmt.Errorf("%w: model %s: %s", ErrUnavailable, model, reason)
		}
		if g.failure != nil {
			err := g.failure
			g.mu.Unlock()
			return ModelCard{}, fmt.Errorf("%w: %w", ErrUnavailable, err)
		}
		state, changed := served.state, g.changed
		g.mu.Unlock()
		select {
		case <-ctx.Done():
			return ModelCard{}, fmt.Errorf("%w: model %s is %s: %w", ErrUnavailable, model, state, ctx.Err())
		case <-changed:
		}
	}
}

func (g *group) status() []DeploymentStatus {
	g.mu.Lock()
	defer g.mu.Unlock()
	artifacts := make(map[string]string, len(g.plan.models))
	for _, entry := range g.plan.models {
		artifacts[entry.Name] = entry.Model
	}
	statuses := make([]DeploymentStatus, 0, len(g.plan.members))
	for _, deployment := range g.plan.deployments() {
		served := g.models[g.plan.members[deployment]]
		status := DeploymentStatus{
			Name: deployment, Managed: g.managed, Endpoint: g.client.Endpoint(), Process: g.plan.name,
			Model: served.name, Artifact: artifacts[served.name], Ready: served.ready.Load(), State: served.state,
			Reason: served.reason, Restarts: g.restarts,
		}
		if served.card != nil {
			card := *served.card
			status.Card = &card
		}
		statuses = append(statuses, status)
	}
	sort.Slice(statuses, func(i, j int) bool { return statuses[i].Name < statuses[j].Name })
	return statuses
}

func boolGauge(value bool) float64 {
	if value {
		return 1
	}
	return 0
}

func (g *group) recordReady(deployment string, ready bool) {
	replicaReadyGauge.WithLabelValues(deployment, g.plan.replica).Set(boolGauge(ready))
	if g.plan.pool {
		return
	}
	readyGauge.WithLabelValues(deployment).Set(boolGauge(ready))
}
