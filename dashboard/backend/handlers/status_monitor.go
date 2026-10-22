package handlers

import (
	"context"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/dashboard/backend/statusstore"
)

const statusObservationInterval = 30 * time.Second

// RouterReadiness is one observation of whether the Router can serve requests.
type RouterReadiness struct {
	Ready     bool
	CheckedAt time.Time
}

// StatusMonitor owns service sampling so browser traffic never determines the
// availability history shown on the public status page. It is also the one
// source of Router readiness: its loop and every /api/status request publish
// what they observe, and readers get the newest observation.
type StatusMonitor struct {
	routerAPIURL       string
	envoyURL           string
	configDir          string
	stack              StackState
	historyStore       *statusstore.Store
	credentialProvider []routerauth.CredentialProvider
	interval           time.Duration
	stop               chan struct{}
	done               chan struct{}
	startOnce          sync.Once
	closeOnce          sync.Once
	readinessMu        sync.RWMutex
	readiness          RouterReadiness
}

func NewStatusMonitor(
	routerAPIURL string,
	envoyURL string,
	configDir string,
	stack StackState,
	historyStore *statusstore.Store,
	credentialProvider ...routerauth.CredentialProvider,
) *StatusMonitor {
	return &StatusMonitor{
		routerAPIURL:       routerAPIURL,
		envoyURL:           envoyURL,
		configDir:          configDir,
		stack:              stack,
		historyStore:       historyStore,
		credentialProvider: credentialProvider,
		interval:           statusObservationInterval,
		stop:               make(chan struct{}),
		done:               make(chan struct{}),
	}
}

func (m *StatusMonitor) Start() {
	m.startOnce.Do(func() { go m.run() })
}

// Close cancels in-flight probes and waits for the loop to stop.
func (m *StatusMonitor) Close() error {
	m.closeOnce.Do(func() { close(m.stop) })
	m.startOnce.Do(func() { close(m.done) })
	select {
	case <-m.done:
		return nil
	case <-time.After(10 * time.Second):
		return context.DeadlineExceeded
	}
}

// RouterReadiness returns the newest observation; ok is false before the
// first one completes.
func (m *StatusMonitor) RouterReadiness() (readiness RouterReadiness, ok bool) {
	m.readinessMu.RLock()
	defer m.readinessMu.RUnlock()
	return m.readiness, !m.readiness.CheckedAt.IsZero()
}

func (m *StatusMonitor) run() {
	defer close(m.done)
	// Close cancels this context so an in-flight probe stops at once.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-m.stop
		cancel()
	}()
	m.sample(ctx)
	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			m.sample(ctx)
		case <-ctx.Done():
			return
		}
	}
}

// observe detects the system status and publishes the Router readiness it
// found. An observation cut short by ctx is not published: a disconnected
// client or a closing monitor says nothing about the Router.
func (m *StatusMonitor) observe(ctx context.Context) SystemStatus {
	startedAt := time.Now().UTC()
	status := detectSystemStatus(ctx, m.routerAPIURL, m.envoyURL, m.configDir, m.stack, m.credentialProvider...)
	if ctx.Err() == nil {
		m.publish(RouterReadiness{Ready: status.routerReady, CheckedAt: startedAt})
	}
	return status
}

// publish keeps the newest observation by start time, so a slow probe that
// began earlier never overwrites a later one.
func (m *StatusMonitor) publish(readiness RouterReadiness) {
	m.readinessMu.Lock()
	defer m.readinessMu.Unlock()
	if readiness.CheckedAt.After(m.readiness.CheckedAt) {
		m.readiness = readiness
	}
}

func (m *StatusMonitor) sample(ctx context.Context) {
	status := m.observe(ctx)
	if m.historyStore == nil || ctx.Err() != nil {
		return
	}
	recordCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := m.historyStore.Record(recordCtx, statusObservations(status.Services)); err != nil {
		log.Printf("status history observation failed: %v", err)
	}
}

func statusObservations(services []ServiceStatus) []statusstore.Observation {
	observations := make([]statusstore.Observation, 0, len(services))
	for _, service := range services {
		state := statusstore.StateUnavailable
		if service.Healthy {
			state = statusstore.StateOperational
		} else if strings.Contains(strings.ToLower(service.Status), "start") ||
			strings.Contains(strings.ToLower(service.Message), "start") {
			state = statusstore.StateStarting
		}
		observations = append(observations, statusstore.Observation{Service: service.Name, State: state})
	}
	return observations
}
