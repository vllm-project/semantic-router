package modelservice

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// DeploymentStatus is the observable state of one deployment. Restarts counts
// the exits of its managed process; an attached endpoint has none.
type DeploymentStatus struct {
	Name     string     `json:"name"`
	Managed  bool       `json:"managed"`
	Endpoint string     `json:"endpoint"`
	Process  string     `json:"process"`
	Model    string     `json:"model"`
	Ready    bool       `json:"ready"`
	State    string     `json:"state"`
	Reason   string     `json:"reason,omitempty"`
	Restarts int        `json:"restarts"`
	Card     *ModelCard `json:"card,omitempty"`
}

// Lease is one router generation's view of its model_runtime deployments.
// It holds a reference to every process the generation uses, so a reload
// with the same composition shares the running processes and one with a new
// composition starts new ones without disturbing the previous generation.
type Lease struct {
	manager   *Manager
	mu        sync.RWMutex
	members   map[string]member
	groups    []*group
	closeOnce sync.Once
}

type member struct {
	group  *group
	served *servedModel
}

// Deployments lists the deployment names the lease serves.
func (l *Lease) Deployments() []string {
	if l == nil {
		return nil
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	names := make([]string, 0, len(l.members))
	for name := range l.members {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Ensure adds a deployment the generation's plan did not include, in a
// process of its own, so a consumer the plan missed still runs. It is a no-op
// for a deployment the lease already serves.
func (l *Lease) Ensure(name string, deployment config.ModelDeployment) error {
	if _, ok := l.lookup(name); ok {
		return nil
	}
	return l.manager.extend(l, name, deployment)
}

// Close releases the lease's processes; the last reference stops a process.
func (l *Lease) Close() error {
	if l == nil {
		return nil
	}
	l.closeOnce.Do(func() {
		l.mu.Lock()
		groups := l.groups
		l.groups = nil
		l.mu.Unlock()
		l.manager.release(groups)
	})
	return nil
}

// Statuses reports the lease's deployments, sorted by name.
func (l *Lease) Statuses() []DeploymentStatus {
	if l == nil {
		return nil
	}
	l.mu.RLock()
	groups := append([]*group(nil), l.groups...)
	l.mu.RUnlock()
	var statuses []DeploymentStatus
	for _, g := range groups {
		for _, status := range g.status() {
			if _, ok := l.lookup(status.Name); ok {
				statuses = append(statuses, status)
			}
		}
	}
	sort.Slice(statuses, func(i, j int) bool { return statuses[i].Name < statuses[j].Name })
	return statuses
}

// Card waits until a deployment is ready and returns its model card.
func (l *Lease) Card(ctx context.Context, deployment string) (ModelCard, error) {
	m, ok := l.lookup(deployment)
	if !ok {
		return ModelCard{}, ErrUnknownDeployment
	}
	return m.group.waitCard(ctx, m.served.name)
}

func (l *Lease) lookup(deployment string) (member, bool) {
	if l == nil {
		return member{}, false
	}
	l.mu.RLock()
	m, ok := l.members[deployment]
	l.mu.RUnlock()
	return m, ok
}

// call resolves a ready deployment; a deployment that is not ready fails at
// once with ErrUnavailable so callers fail open without waiting.
func (l *Lease) call(deployment string) (member, error) {
	m, ok := l.lookup(deployment)
	if !ok {
		requestsTotal.WithLabelValues(deployment, ErrorReason(ErrUnknownDeployment)).Inc()
		return member{}, ErrUnknownDeployment
	}
	if !m.served.ready.Load() {
		requestsTotal.WithLabelValues(deployment, ErrorReason(ErrUnavailable)).Inc()
		return member{}, ErrUnavailable
	}
	return m, nil
}

func (l *Lease) observe(m member, deployment, surface string, started time.Time, err error) {
	requestDuration.WithLabelValues(deployment, surface).Observe(time.Since(started).Seconds())
	requestsTotal.WithLabelValues(deployment, ErrorReason(err)).Inc()
	if errors.Is(err, ErrFailed) {
		// A transport failure may mean the process died: probe now rather
		// than at the next slow poll.
		m.group.requestProbe()
	}
}

// Decide answers the request through a deployment, inside the context's
// bundle when there is one; the bundle may answer it together with the
// stage's other questions to the same model and state.
func (l *Lease) Decide(ctx context.Context, deployment string, request Request) (Response, error) {
	m, err := l.call(deployment)
	if err != nil {
		return Response{}, err
	}
	cache := m.served.cache.active()
	if InBundle(ctx) {
		started := time.Now()
		request.Model = m.served.name
		response, decideErr := m.group.client.decide(ctx, request, cache, deployment)
		l.observe(m, deployment, "decisions", started, decideErr)
		return response, decideErr
	}
	var key cacheKey
	if cache != nil {
		key = decideKey(request)
		if cached, ok := cache.get(key); ok {
			cacheTotal.WithLabelValues(deployment, "hit").Inc()
			return cached.(Response), nil
		}
	}
	started := time.Now()
	request.Model = m.served.name
	response, err := m.group.client.Decide(ctx, request)
	l.observe(m, deployment, "decisions", started, err)
	if err == nil && cache != nil {
		cacheTotal.WithLabelValues(deployment, "miss").Inc()
		if complete(response) {
			cache.put(key, response)
		}
	}
	return response, err
}

// Classify runs a classify request on a deployment, inside the context's bundle when there is one.
func (l *Lease) Classify(ctx context.Context, deployment string, request ClassifyRequest) (ClassifyResponse, error) {
	m, err := l.call(deployment)
	if err != nil {
		return ClassifyResponse{}, err
	}
	cache := m.served.cache.active()
	var key cacheKey
	if cache != nil {
		key = classifyKey(request)
		if cached, ok := cache.get(key); ok {
			cacheTotal.WithLabelValues(deployment, "hit").Inc()
			return cached.(ClassifyResponse), nil
		}
	}
	started := time.Now()
	response, err := m.group.client.Classify(ctx, m.served.name, request)
	l.observe(m, deployment, "classify", started, err)
	if err == nil && cache != nil {
		cacheTotal.WithLabelValues(deployment, "miss").Inc()
		if classified(response) {
			cache.put(key, response)
		}
	}
	return response, err
}

// Embed runs an embeddings request on a deployment, inside the context's bundle when there is one.
func (l *Lease) Embed(ctx context.Context, deployment string, request EmbedRequest) (EmbedResponse, error) {
	m, err := l.call(deployment)
	if err != nil {
		return EmbedResponse{}, err
	}
	started := time.Now()
	response, err := m.group.client.Embed(ctx, m.served.name, request)
	l.observe(m, deployment, "embeddings", started, err)
	return response, err
}

// Rerank runs a rerank request on a deployment, inside the context's bundle when there is one.
func (l *Lease) Rerank(ctx context.Context, deployment string, request RerankRequest) (RerankResponse, error) {
	m, err := l.call(deployment)
	if err != nil {
		return RerankResponse{}, err
	}
	started := time.Now()
	response, err := m.group.client.Rerank(ctx, m.served.name, request)
	l.observe(m, deployment, "rerank", started, err)
	return response, err
}

// classified reports whether every input has a result, so the response may be cached.
func classified(response ClassifyResponse) bool {
	for _, result := range response.Results {
		if result.Error != "" {
			return false
		}
	}
	return len(response.Results) > 0
}

func complete(response Response) bool {
	for _, answer := range response.Answers {
		if answer.Error != "" {
			return false
		}
	}
	return len(response.Answers) > 0
}
