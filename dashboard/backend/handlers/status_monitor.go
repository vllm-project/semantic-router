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

const statusObservationInterval = time.Minute

// StatusMonitor owns service sampling so browser traffic never determines the
// availability history shown on the public status page.
type StatusMonitor struct {
	routerAPIURL       string
	envoyURL           string
	configDir          string
	historyStore       *statusstore.Store
	credentialProvider []routerauth.CredentialProvider
	interval           time.Duration
	stop               chan struct{}
	done               chan struct{}
	startOnce          sync.Once
	closeOnce          sync.Once
}

func NewStatusMonitor(
	routerAPIURL string,
	envoyURL string,
	configDir string,
	historyStore *statusstore.Store,
	credentialProvider ...routerauth.CredentialProvider,
) *StatusMonitor {
	return &StatusMonitor{
		routerAPIURL:       routerAPIURL,
		envoyURL:           envoyURL,
		configDir:          configDir,
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

func (m *StatusMonitor) Close() error {
	m.closeOnce.Do(func() { close(m.stop) })
	select {
	case <-m.done:
		return nil
	case <-time.After(10 * time.Second):
		return context.DeadlineExceeded
	}
}

func (m *StatusMonitor) run() {
	defer close(m.done)
	m.sample()
	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			m.sample()
		case <-m.stop:
			return
		}
	}
}

func (m *StatusMonitor) sample() {
	if m.historyStore == nil {
		return
	}
	status := detectSystemStatus(m.routerAPIURL, m.envoyURL, m.configDir, m.credentialProvider...)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := m.historyStore.Record(ctx, statusObservations(status.Services)); err != nil {
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
