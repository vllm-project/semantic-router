package modelservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

const SystemOneResponseLimit = 4 << 20

var ErrSystemOneArtifactChanged = errors.New("published model artifact differs from requested identity")

// SystemOneResult preserves the native System One response, including per-question
// errors, probabilities, usage, spans and metadata that routing adapters discard.
type SystemOneResult struct {
	Status       int
	Body         json.RawMessage
	ServerTiming string
}

// SystemOne invokes only a deployment in the published configuration. It retains
// that process across a config reload and never interprets a browser-supplied URL.
// Probes bypass the Router result cache so each run measures actual inference.
func (m *Manager) SystemOne(ctx context.Context, deployment string, body json.RawMessage) (SystemOneResult, error) {
	return m.SystemOneForArtifact(ctx, deployment, "", body)
}

// SystemOneForArtifact pins publication and artifact admission under the same
// lock. Saving a new resource under an existing key cannot call the old model
// through the new public identity while its deployment is still pending.
func (m *Manager) SystemOneForArtifact(ctx context.Context, deployment, expectedArtifact string, body json.RawMessage) (SystemOneResult, error) {
	m.mu.Lock()
	lease := m.published
	if lease == nil || m.closed {
		m.mu.Unlock()
		return SystemOneResult{}, ErrUnavailable
	}
	selected, err := lease.call(deployment)
	if err != nil {
		m.mu.Unlock()
		return SystemOneResult{}, err
	}
	if expectedArtifact != "" {
		artifact := ""
		if selected.pool != nil {
			artifact = selected.pool.declaration.Artifact
			if !selected.pool.declaration.Managed() {
				selected.pool.mu.Lock()
				artifact = ""
				if selected.pool.baseline != nil {
					artifact = selected.pool.baseline.Repo
				}
				selected.pool.mu.Unlock()
			}
		}
		var entries []modelEntry
		if selected.pool == nil {
			entries = selected.group.plan.models
		}
		for _, entry := range entries {
			if entry.Name == selected.served.name {
				artifact = entry.Model
				break
			}
		}
		if artifact == "" && selected.pool == nil && !selected.group.managed {
			selected.group.mu.Lock()
			if selected.served.card != nil {
				artifact = selected.served.card.Repo
			}
			selected.group.mu.Unlock()
		}
		if artifact != expectedArtifact {
			m.mu.Unlock()
			return SystemOneResult{}, ErrSystemOneArtifactChanged
		}
	}
	retained := retainMemberLocked(selected)
	m.mu.Unlock()
	defer m.release(retained)
	started := time.Now()
	result, timing, err := selected.client().systemOne(ctx, selected.served.name, body)
	lease.observe(selected, deployment, "decisions", started, timing, err)
	return result, err
}

func (c *Client) systemOne(ctx context.Context, model string, body json.RawMessage) (SystemOneResult, exchangeTiming, error) {
	// Keep state, questions and options as raw JSON: question and criteria order
	// belongs to the model contract. Only the served model identity is replaced.
	encoded, ok := withServedModel(body, model)
	if !ok {
		var request map[string]json.RawMessage
		if json.Unmarshal(body, &request) != nil || request == nil {
			return SystemOneResult{}, exchangeTiming{}, fmt.Errorf("%w: request must be an object", ErrRejected)
		}
		request["model"], _ = json.Marshal(model)
		var err error
		if encoded, err = json.Marshal(request); err != nil {
			return SystemOneResult{}, exchangeTiming{}, err
		}
	}
	generated, err := api.NewClient(c.base, api.WithHTTPClient(c.httpClient))
	if err != nil {
		return SystemOneResult{}, exchangeTiming{}, err
	}
	started := time.Now()
	response, err := generated.CreateSystemOneWithBody(ctx, "application/json", bytes.NewReader(encoded))
	timing := timedExchange(started, response)
	if err != nil {
		return SystemOneResult{}, timing, fmt.Errorf("%w: %w", ErrFailed, err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, SystemOneResponseLimit+1))
	timing = timedExchange(started, response)
	if err != nil {
		return SystemOneResult{}, timing, fmt.Errorf("%w: %w", ErrFailed, err)
	}
	if len(data) > SystemOneResponseLimit || !json.Valid(data) || response.StatusCode < 200 || response.StatusCode >= 300 && response.StatusCode < 400 {
		return SystemOneResult{}, timing, fmt.Errorf("%w: invalid System One response", ErrFailed)
	}
	result := SystemOneResult{Status: response.StatusCode, Body: data, ServerTiming: response.Header.Get("Server-Timing")}
	return result, timing, systemOneStatusError(response.StatusCode)
}

func systemOneStatusError(status int) error {
	switch {
	case status >= 200 && status < 300:
		return nil
	case status == http.StatusTooManyRequests:
		return ErrOverloaded
	case status == http.StatusServiceUnavailable:
		return ErrUnavailable
	case status >= 400 && status < 500:
		return ErrRejected
	default:
		return ErrFailed
	}
}
