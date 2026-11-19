package handlers

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	modelHealthInterval = 30 * time.Second
	// A check younger than modelHealthFreshFor answers every viewer; a
	// refresh still waits modelHealthRefreshFloor between checks of a model.
	modelHealthFreshFor     = 20 * time.Second
	modelHealthRefreshFloor = 5 * time.Second
	modelHealthProbeTimeout = 5 * time.Second
	// modelHealthWait bounds how long a request waits for a round; a longer
	// round keeps running and its results answer the next request.
	modelHealthWait        = 8 * time.Second
	modelHealthSlowAfter   = 2500 * time.Millisecond
	modelHealthHistory     = 30
	modelHealthConcurrency = 16
)

type modelHealthOptions struct {
	client     modelVerificationHTTPClient
	loadConfig func(string) (*routerconfig.RouterConfig, error)
	now        func() time.Time
	timeout    time.Duration
	wait       time.Duration
}

type modelHealthService struct {
	configPath string
	client     modelVerificationHTTPClient
	loadConfig func(string) (*routerconfig.RouterConfig, error)
	now        func() time.Time
	timeout    time.Duration
	wait       time.Duration
	// round admits one probe round at a time, so concurrent viewers share it.
	round chan struct{}
	mu    sync.Mutex
	state map[string]*modelHealthState
}

type modelHealthState struct {
	fingerprint string
	checked     time.Time
	entry       ModelHealthEntry
	samples     []int64
}

var modelHealthDefaultClient = &http.Client{
	CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	},
}

func newModelHealthService(configPath string, options modelHealthOptions) *modelHealthService {
	loadConfig := options.loadConfig
	if loadConfig == nil {
		loadConfig = ReadPersistedRouterConfig
	}
	now := options.now
	if now == nil {
		now = time.Now
	}
	timeout := options.timeout
	if timeout <= 0 || timeout > modelHealthProbeTimeout {
		timeout = modelHealthProbeTimeout
	}
	wait := options.wait
	if wait <= 0 || wait > modelHealthWait {
		wait = modelHealthWait
	}
	return &modelHealthService{
		configPath: strings.TrimSpace(configPath),
		client:     options.client,
		loadConfig: loadConfig,
		now:        now,
		timeout:    timeout,
		wait:       wait,
		round:      make(chan struct{}, 1),
		state:      make(map[string]*modelHealthState),
	}
}

// Check returns the health of the named models, or of every configured model
// when none is named, checking again those whose last check is stale.
func (service *modelHealthService) Check(ctx context.Context, models []string, refresh bool) (ModelHealthResponse, error) {
	plans, err := service.plans(models)
	if err != nil {
		return ModelHealthResponse{}, err
	}
	service.refreshStale(ctx, plans, refresh)
	if len(models) == 0 {
		service.prune(plans)
	}
	return service.response(plans), nil
}

func (service *modelHealthService) plans(models []string) ([]modelHealthPlan, error) {
	if service.configPath == "" {
		return nil, modelVerificationFailure(http.StatusServiceUnavailable, "config_unavailable", "Active model configuration is unavailable.")
	}
	config, err := service.loadConfig(service.configPath)
	if err != nil || config == nil {
		return nil, modelVerificationFailure(http.StatusServiceUnavailable, "config_unavailable", "Active model configuration could not be loaded.")
	}
	names := models
	if len(names) == 0 {
		names = make([]string, 0, len(config.ModelConfig))
		for name := range config.ModelConfig {
			names = append(names, name)
		}
		sort.Strings(names)
	}
	// The catalog only adds list paths and default headers; without it a
	// backend is still checked through its health path or base URL.
	registry, _ := modelcatalog.BuiltIn()
	plans := make([]modelHealthPlan, 0, len(names))
	for _, name := range names {
		if _, configured := config.ModelConfig[name]; !configured {
			return nil, modelVerificationFailure(http.StatusNotFound, "model_not_found", fmt.Sprintf("Model %q is not present in the active provider configuration.", name))
		}
		plans = append(plans, resolveModelHealthPlan(config, registry, name))
	}
	return plans, nil
}

func (service *modelHealthService) stale(plans []modelHealthPlan, refresh bool) []modelHealthPlan {
	maxAge := modelHealthFreshFor
	if refresh {
		maxAge = modelHealthRefreshFloor
	}
	now := service.now()
	service.mu.Lock()
	defer service.mu.Unlock()
	var stale []modelHealthPlan
	for _, plan := range plans {
		state := service.state[plan.model]
		if state == nil || state.fingerprint != plan.fingerprint || now.Sub(state.checked) >= maxAge {
			stale = append(stale, plan)
		}
	}
	return stale
}

func (service *modelHealthService) refreshStale(ctx context.Context, plans []modelHealthPlan, refresh bool) {
	if len(service.stale(plans, refresh)) == 0 {
		return
	}
	timer := time.NewTimer(service.wait)
	defer timer.Stop()
	select {
	case service.round <- struct{}{}:
	case <-ctx.Done():
		return
	case <-timer.C:
		return
	}
	// The round that held the slot may have checked these models meanwhile.
	stale := service.stale(plans, refresh)
	if len(stale) == 0 {
		<-service.round
		return
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer func() { <-service.round }()
		service.probeRound(context.WithoutCancel(ctx), stale)
	}()
	select {
	case <-done:
	case <-ctx.Done():
	case <-timer.C:
	}
}

// probeRound checks every backend of the plans, at most
// modelHealthConcurrency at a time, and records each model when its
// backends have answered.
func (service *modelHealthService) probeRound(ctx context.Context, plans []modelHealthPlan) {
	limit := make(chan struct{}, modelHealthConcurrency)
	var models sync.WaitGroup
	for _, plan := range plans {
		models.Add(1)
		go func() {
			defer models.Done()
			probes := make([]modelHealthProbe, len(plan.targets))
			var targets sync.WaitGroup
			for index, target := range plan.targets {
				targets.Add(1)
				go func() {
					defer targets.Done()
					limit <- struct{}{}
					defer func() { <-limit }()
					probes[index] = service.probe(ctx, target)
				}()
			}
			targets.Wait()
			service.record(plan, probes)
		}()
	}
	models.Wait()
}

func (service *modelHealthService) record(plan modelHealthPlan, probes []modelHealthProbe) {
	checked := service.now()
	entry := aggregateModelHealth(plan, probes)
	entry.CheckedAt = checked.UTC().Format(time.RFC3339)

	service.mu.Lock()
	defer service.mu.Unlock()
	state := service.state[plan.model]
	if state == nil || state.fingerprint != plan.fingerprint {
		state = &modelHealthState{fingerprint: plan.fingerprint}
		service.state[plan.model] = state
	}
	for _, probe := range probes {
		if probe.answered {
			state.samples = append(state.samples, probe.latencyMS)
		}
	}
	if extra := len(state.samples) - modelHealthHistory; extra > 0 {
		state.samples = append([]int64(nil), state.samples[extra:]...)
	}
	if len(state.samples) > 0 && entry.Status != ModelHealthStatusUnchecked {
		entry.RecentMS = append([]int64(nil), state.samples...)
		p95 := percentile95(state.samples)
		entry.P95MS = &p95
	}
	state.entry = entry
	state.checked = checked
}

func (service *modelHealthService) prune(plans []modelHealthPlan) {
	configured := make(map[string]struct{}, len(plans))
	for _, plan := range plans {
		configured[plan.model] = struct{}{}
	}
	service.mu.Lock()
	defer service.mu.Unlock()
	for model := range service.state {
		if _, found := configured[model]; !found {
			delete(service.state, model)
		}
	}
}

func (service *modelHealthService) response(plans []modelHealthPlan) ModelHealthResponse {
	service.mu.Lock()
	defer service.mu.Unlock()
	var latest time.Time
	models := make([]ModelHealthEntry, 0, len(plans))
	for _, plan := range plans {
		state := service.state[plan.model]
		if state == nil || state.fingerprint != plan.fingerprint {
			models = append(models, ModelHealthEntry{
				Model:  plan.model,
				Status: ModelHealthStatusUnchecked,
				Reason: "The first check is still running.",
			})
			continue
		}
		if state.checked.After(latest) {
			latest = state.checked
		}
		models = append(models, cloneModelHealthEntry(state.entry))
	}
	if latest.IsZero() {
		latest = service.now()
	}
	return ModelHealthResponse{
		CheckedAt:       latest.UTC().Format(time.RFC3339),
		IntervalSeconds: int(modelHealthInterval / time.Second),
		Models:          models,
	}
}

func cloneModelHealthEntry(entry ModelHealthEntry) ModelHealthEntry {
	entry.RecentMS = append([]int64(nil), entry.RecentMS...)
	entry.Endpoints = append([]ModelHealthEndpoint(nil), entry.Endpoints...)
	return entry
}

// percentile95 is the nearest-rank 95th percentile of the samples.
func percentile95(samples []int64) int64 {
	if len(samples) == 0 {
		return 0
	}
	sorted := append([]int64(nil), samples...)
	sort.Slice(sorted, func(left, right int) bool { return sorted[left] < sorted[right] })
	rank := (len(sorted)*95 + 99) / 100
	return sorted[rank-1]
}
