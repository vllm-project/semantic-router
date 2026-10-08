package modelservice

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// ReplicaStatus describes local admission and readiness. It deliberately omits
// endpoint addresses. EstimatedWork is outstanding request bytes, not remote
// queue telemetry or a measured token count.
type ReplicaStatus struct {
	ID            string `json:"id"`
	Managed       bool   `json:"managed"`
	Device        string `json:"device,omitempty"`
	Ready         bool   `json:"ready"`
	State         string `json:"state"`
	Reason        string `json:"reason,omitempty"`
	Inflight      int    `json:"inflight"`
	EstimatedWork int64  `json:"estimated_work"`
	Restarts      int    `json:"restarts"`
}

const replicaAdmissionLimit = 32

type replicaOutcome string

const (
	replicaOK       replicaOutcome = "ok"
	replicaFailed   replicaOutcome = "failed"
	replicaRejected replicaOutcome = "rejected"
	replicaCanceled replicaOutcome = "canceled"
	replicaTimeout  replicaOutcome = "timeout"
)

// replicaLoad belongs to a physical worker, so overlapping generations and
// native/Router leases observe the same outstanding work and health backoff.
type replicaLoad struct {
	mu       sync.Mutex
	inflight int
	work     int64
	backoff  time.Time
	failures int
	// Protected by the manager lock, shared across overlapping generations.
	lastAssigned uint64
}

type replicaWorker struct {
	id     string
	member member
}
type replicaPool struct {
	manager     *Manager
	declaration config.ModelDeployment
	name        string
	workers     []replicaWorker
	client      *Client
	served      *servedModel
	mu          sync.Mutex
	baseline    *ModelCard
}

func newReplicaPool(manager *Manager, name string, declaration config.ModelDeployment, workers []replicaWorker) *replicaPool {
	p := &replicaPool{manager: manager, name: name, declaration: declaration.WithDefaults(), workers: workers}
	generated, _ := api.NewClientWithResponses("http://model-runtime", api.WithHTTPClient(p))
	p.client = &Client{endpoint: "pool", base: "http://model-runtime", httpClient: p, api: generated}
	p.client.bundleTasks.Store(DefaultBundleTasks)
	p.client.apiMinor.Store(-1)
	// A single worker reuses its existing result cache. Multi-worker output
	// equivalence is not inferred from structural capabilities, so those pools
	// bypass the Router result cache.
	p.served = &servedModel{name: name, cache: newResultCache(0)}
	if len(workers) == 1 {
		p.served.cache = workers[0].member.served.cache
	}
	return p
}

func (p *replicaPool) workerCard(w replicaWorker) (ModelCard, error) {
	g, served := w.member.group, w.member.served
	g.mu.Lock()
	defer g.mu.Unlock()
	if !served.ready.Load() || served.card == nil {
		return ModelCard{}, ErrUnavailable
	}
	card := *served.card
	if len(p.workers) > 1 {
		if !g.managed && card.Repo != p.declaration.Artifact {
			return ModelCard{}, fmt.Errorf("replica artifact does not match deployment")
		}
		if p.declaration.Revision != "" && card.Revision != p.declaration.Revision {
			return ModelCard{}, fmt.Errorf("replica revision does not match deployment")
		}
		if card.Profile != p.declaration.Profile {
			return ModelCard{}, fmt.Errorf("replica profile does not match deployment")
		}
		if card.ModelSHA256 == "" {
			return ModelCard{}, fmt.Errorf("replica does not report model identity")
		}
	}
	return card, nil
}

func comparableReplicaCard(card ModelCard) ModelCard {
	card.ID, card.Device, card.Accelerator, card.Engine, card.Dtype, card.Status, card.Reason = "", "", "", "", "", "", ""
	card.Ready = false
	return card
}

// readyWorkers validates every ready observation before it can enter admission.
// A changed or heterogeneous worker never silently receives a request.
func (p *replicaPool) readyWorkers() ([]replicaWorker, map[string]string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	var ready []replicaWorker
	reasons := make(map[string]string)
	limit, minor := int64(0), int64(-1)
	for _, w := range p.workers {
		card, err := p.workerCard(w)
		if err != nil {
			reasons[w.id] = err.Error()
			continue
		}
		comparable := comparableReplicaCard(card)
		if p.baseline == nil || len(p.workers) == 1 {
			p.baseline = &card
		}
		if !reflect.DeepEqual(comparableReplicaCard(*p.baseline), comparable) {
			reasons[w.id] = "replica identity or capabilities differ from pool"
			continue
		}
		load := &w.member.served.load
		load.mu.Lock()
		cooling := time.Now().Before(load.backoff)
		load.mu.Unlock()
		if cooling {
			reasons[w.id] = "replica is in health backoff"
			continue
		}
		ready = append(ready, w)
		current := w.member.group.client.bundleTasks.Load()
		if limit == 0 || current < limit {
			limit = current
		}
		current = w.member.group.client.apiMinor.Load()
		if minor < 0 || current < minor {
			minor = current
		}
	}
	if len(ready) > 0 {
		p.client.bundleTasks.Store(limit)
		p.client.apiMinor.Store(minor)
		inputs := map[string]int{p.name: p.baseline.MaxInputs}
		scanning := map[string]bool{p.name: p.baseline.MaxScanTokens > 0}
		p.client.maxInputs.Store(&inputs)
		p.client.scanning.Store(&scanning)
	}
	p.served.ready.Store(len(ready) > 0)
	readyGauge.WithLabelValues(p.name).Set(boolGauge(len(ready) > 0))
	return ready, reasons
}

func (p *replicaPool) card(ctx context.Context) (ModelCard, error) {
	if len(p.workers) == 1 {
		worker := p.workers[0]
		if _, err := worker.member.group.waitCard(ctx, worker.member.served.name); err != nil {
			return ModelCard{}, err
		}
	}
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	for {
		ready, reasons := p.readyWorkers()
		if len(ready) > 0 {
			p.mu.Lock()
			card := *p.baseline
			p.mu.Unlock()
			card.ID, card.Ready, card.Status = p.name, true, "ready"
			return card, nil
		}
		allObserved := true
		var failures []error
		for _, w := range p.workers {
			g, served := w.member.group, w.member.served
			g.mu.Lock()
			if !served.ready.Load() {
				allObserved = false
			}
			if err := g.cardFailureLocked(served); err != nil {
				failures = append(failures, err)
			} else if served.ready.Load() && reasons[w.id] != "" {
				failures = append(failures, fmt.Errorf("%w: %s", ErrUnavailable, reasons[w.id]))
			}
			g.mu.Unlock()
		}
		if len(failures) == len(p.workers) {
			return ModelCard{}, fmt.Errorf("no replica can become ready: %w", errors.Join(failures...))
		}
		if allObserved {
			return ModelCard{}, fmt.Errorf("%w: no compatible replica (%v)", ErrUnavailable, reasons)
		}
		select {
		case <-ctx.Done():
			return ModelCard{}, fmt.Errorf("%w: %w", ErrUnavailable, ctx.Err())
		case <-ticker.C:
		}
	}
}

func (p *replicaPool) status() DeploymentStatus {
	ready, reasons := p.readyWorkers()
	status := DeploymentStatus{Name: p.name, Model: p.name, Artifact: p.declaration.Artifact, DesiredReplicas: len(p.workers), ReadyReplicas: len(ready), Ready: len(ready) > 0, State: "unavailable"}
	if len(p.workers) == 1 {
		worker := p.workers[0]
		status.Model = worker.member.served.name
		status.Endpoint = worker.member.group.client.Endpoint()
		status.Process = worker.member.group.plan.name
		if len(ready) == 0 {
			physical := worker.member.group.status()[0]
			status.State = physical.State
			status.Reason = physical.Reason
			if reasons[worker.id] == "replica is in health backoff" {
				status.State = "backoff"
				status.Reason = reasons[worker.id]
			}
		}
	}
	if len(ready) > 0 {
		status.State = "ready"
		if len(ready) < len(p.workers) {
			status.State = "degraded"
		}
		p.mu.Lock()
		card := *p.baseline
		p.mu.Unlock()
		card.ID, card.Ready = p.name, true
		status.Card = &card
	}
	for _, w := range p.workers {
		g := w.member.group
		g.mu.Lock()
		replica := ReplicaStatus{ID: w.id, Managed: g.managed, Ready: w.member.served.ready.Load() && reasons[w.id] == "", State: w.member.served.state, Reason: w.member.served.reason, Restarts: g.restarts}
		if w.member.served.card != nil {
			replica.Device = w.member.served.card.Device
		}
		g.mu.Unlock()
		if reason := reasons[w.id]; reason != "" {
			if replica.Reason == "" || w.member.served.ready.Load() {
				replica.Reason = reason
			}
			if w.member.served.ready.Load() {
				if reason == "replica is in health backoff" {
					replica.State = "backoff"
				} else {
					replica.State = "incompatible"
				}
			}
		}
		load := &w.member.served.load
		load.mu.Lock()
		replica.Inflight, replica.EstimatedWork = load.inflight, load.work
		if time.Now().Before(load.backoff) {
			replica.Ready = false
			replica.State = "backoff"
		}
		load.mu.Unlock()
		status.Managed = status.Managed || g.managed
		status.Restarts += replica.Restarts
		status.Replicas = append(status.Replicas, replica)
	}
	// Aggregate the same worker observations returned to callers, rather than a
	// second readiness sample that could disagree after a transport failure.
	status.ReadyReplicas = 0
	for _, replica := range status.Replicas {
		if replica.Ready {
			status.ReadyReplicas++
		}
	}
	status.Ready = status.ReadyReplicas > 0
	if status.Ready {
		status.State = "ready"
		if status.ReadyReplicas < status.DesiredReplicas {
			status.State = "degraded"
		}
	} else if len(status.Replicas) == 1 {
		status.State = status.Replicas[0].State
		status.Reason = status.Replicas[0].Reason
	} else {
		status.State = "unavailable"
	}
	return status
}

// retain selects one worker for an already fused exchange. The physical
// reference lasts through response consumption, so closing old generations
// drains accepted work before stopping an otherwise unused managed process.
func (p *replicaPool) retain(cost int64) (replicaWorker, func(replicaOutcome), error) {
	ready, _ := p.readyWorkers()
	p.manager.mu.Lock()
	defer p.manager.mu.Unlock()
	if p.manager.closed {
		return replicaWorker{}, nil, ErrUnavailable
	}
	var chosen *replicaWorker
	best := int64(0)
	var oldest uint64
	for i := range ready {
		w := &ready[i]
		g := w.member.group
		if g.refs <= 0 {
			continue
		}
		load := &w.member.served.load
		load.mu.Lock()
		available := load.inflight < replicaAdmissionLimit && !time.Now().Before(load.backoff)
		score := load.work
		load.mu.Unlock()
		// An idle or equally loaded pool rotates through physical workers.
		// Readiness filtering and a new configuration generation cannot reset it.
		if available && (chosen == nil || score < best || score == best && load.lastAssigned < oldest) {
			chosen = w
			best = score
			oldest = load.lastAssigned
		}
	}
	if chosen == nil {
		if len(ready) > 0 {
			return replicaWorker{}, nil, ErrOverloaded
		}
		return replicaWorker{}, nil, ErrUnavailable
	}
	w := *chosen
	load := &w.member.served.load
	p.manager.dispatchSequence++
	load.lastAssigned = p.manager.dispatchSequence
	load.mu.Lock()
	load.inflight++
	load.work += cost
	replicaInflightGauge.WithLabelValues(p.name, w.id).Set(float64(load.inflight))
	replicaWorkGauge.WithLabelValues(p.name, w.id).Set(float64(load.work))
	load.mu.Unlock()
	w.member.group.refs++
	var once sync.Once
	release := func(outcome replicaOutcome) {
		once.Do(func() {
			load.mu.Lock()
			load.inflight--
			load.work -= cost
			replicaInflightGauge.WithLabelValues(p.name, w.id).Set(float64(load.inflight))
			replicaWorkGauge.WithLabelValues(p.name, w.id).Set(float64(load.work))
			replicaRequestsTotal.WithLabelValues(p.name, w.id, string(outcome)).Inc()
			if outcome == replicaFailed {
				load.failures++
				delay := time.Duration(1<<min(load.failures-1, 5)) * 100 * time.Millisecond
				load.backoff = time.Now().Add(delay)
			} else if outcome == replicaOK {
				load.failures = 0
				load.backoff = time.Time{}
			}
			load.mu.Unlock()
			if outcome == replicaFailed {
				w.member.group.requestProbe()
			}
			p.manager.release([]*group{w.member.group})
		})
	}
	return w, release, nil
}

// poolMember preserves the existing client/bundle path; the virtual transport
// picks a worker only when Bundle has completed question and state fusion.
func (p *replicaPool) poolMember() member {
	return member{pool: p, served: p.served, group: p.workers[0].member.group}
}

func (l *Lease) assemblePools(declarations map[string]config.ModelDeployment) {
	for name, declaration := range declarations {
		var workers []replicaWorker
		for _, g := range l.groups {
			if g.plan.logical != name {
				continue
			}
			for internal, served := range g.plan.members {
				workers = append(workers, replicaWorker{id: g.plan.replica, member: member{group: g, served: g.models[served]}})
				delete(l.members, internal)
			}
		}
		l.members[name] = newReplicaPool(l.manager, name, declaration, workers).poolMember()
	}
}
