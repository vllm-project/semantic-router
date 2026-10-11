package handlers

import (
	"context"
	"maps"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
)

const (
	probeRouter    = "router"
	probeSRBench   = "sr-bench"
	probeMLService = "ml-service"

	capabilityProbeInterval = 30 * time.Second
	capabilityProbeTimeout  = 2 * time.Second
)

type probeResult struct {
	ok        bool
	checkedAt time.Time
}

type capabilityProbe struct {
	name  string
	check func(context.Context) bool
}

// CapabilityProber refreshes dependency health in the background so capability
// reads never wait on an upstream.
type CapabilityProber struct {
	probes    []capabilityProbe
	interval  time.Duration
	mu        sync.RWMutex
	results   map[string]probeResult
	stop      chan struct{}
	done      chan struct{}
	startOnce sync.Once
	closeOnce sync.Once
}

// NewCapabilityProber probes the Router management API, the sr-bench service,
// and the ML service when it runs as a sidecar.
func NewCapabilityProber(cfg *config.Config, credentialProvider ...routerauth.CredentialProvider) *CapabilityProber {
	probes := []capabilityProbe{{
		name: probeRouter,
		check: func(context.Context) bool {
			return checkRouterManagementHealth(cfg.RouterAPIURL+"/ready", credentialProvider...)
		},
	}, {
		name: probeSRBench,
		check: func(ctx context.Context) bool {
			token := strings.TrimSpace(os.Getenv(cfg.SRBenchTokenEnv))
			return token != "" && probeHTTP(ctx, cfg.SRBenchURL+SRBenchAPIPath+"/health", token)
		},
	}}
	if cfg.MLPipelineEnabled && cfg.MLServiceURL != "" {
		probes = append(probes, capabilityProbe{
			name: probeMLService,
			check: func(ctx context.Context) bool {
				return probeHTTP(ctx, strings.TrimRight(cfg.MLServiceURL, "/")+"/api/health", "")
			},
		})
	}
	return newCapabilityProber(probes, capabilityProbeInterval)
}

func newCapabilityProber(probes []capabilityProbe, interval time.Duration) *CapabilityProber {
	return &CapabilityProber{
		probes:   probes,
		interval: interval,
		results:  map[string]probeResult{},
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
	}
}

func (p *CapabilityProber) Start() {
	p.startOnce.Do(func() { go p.run() })
}

func (p *CapabilityProber) Close() error {
	p.closeOnce.Do(func() { close(p.stop) })
	p.startOnce.Do(func() { close(p.done) })
	select {
	case <-p.done:
		return nil
	case <-time.After(10 * time.Second):
		return context.DeadlineExceeded
	}
}

// Snapshot returns the latest result per probe. A probe absent from the map
// has not completed yet.
func (p *CapabilityProber) Snapshot() map[string]probeResult {
	if p == nil {
		return nil
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	return maps.Clone(p.results)
}

func (p *CapabilityProber) run() {
	defer close(p.done)
	p.sample()
	ticker := time.NewTicker(p.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			p.sample()
		case <-p.stop:
			return
		}
	}
}

func (p *CapabilityProber) sample() {
	var wg sync.WaitGroup
	for _, probe := range p.probes {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), capabilityProbeTimeout)
			defer cancel()
			ok := probe.check(ctx)
			p.mu.Lock()
			p.results[probe.name] = probeResult{ok: ok, checkedAt: time.Now().UTC()}
			p.mu.Unlock()
		}()
	}
	wg.Wait()
}

// probeHTTP reports whether url answers 2xx. Redirects are not followed so a
// bearer token never leaves the configured origin.
func probeHTTP(ctx context.Context, url, bearer string) bool {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	if bearer != "" {
		request.Header.Set("Authorization", "Bearer "+bearer)
	}
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	response, err := client.Do(request)
	if err != nil {
		return false
	}
	_ = response.Body.Close()
	return response.StatusCode >= 200 && response.StatusCode < 300
}
