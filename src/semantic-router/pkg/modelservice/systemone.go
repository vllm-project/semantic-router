package modelservice

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

const SystemOneResponseLimit = 4 << 20

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
	m.mu.Lock()
	lease := m.published
	if lease == nil || m.closed {
		m.mu.Unlock()
		return SystemOneResult{}, ErrUnavailable
	}
	member, err := lease.call(deployment)
	if err != nil {
		m.mu.Unlock()
		return SystemOneResult{}, err
	}
	member.group.refs++
	m.mu.Unlock()
	defer m.release([]*group{member.group})
	started := time.Now()
	result, timing, err := member.group.client.systemOne(ctx, member.served.name, body)
	lease.observe(member, deployment, "decisions", started, timing, err)
	return result, err
}

func (c *Client) systemOne(ctx context.Context, model string, body json.RawMessage) (SystemOneResult, exchangeTiming, error) {
	var request map[string]json.RawMessage
	if json.Unmarshal(body, &request) != nil || request == nil {
		return SystemOneResult{}, exchangeTiming{}, fmt.Errorf("%w: request must be an object", ErrRejected)
	}
	// Keep state, questions and options as raw JSON: question and criteria order
	// belongs to the model contract. Only the served model identity is replaced.
	request["model"], _ = json.Marshal(model)
	encoded, err := json.Marshal(request)
	if err != nil {
		return SystemOneResult{}, exchangeTiming{}, err
	}
	base, client, err := newHTTPClient(c.endpoint)
	if err != nil {
		return SystemOneResult{}, exchangeTiming{}, err
	}
	defer client.CloseIdleConnections()
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	generated, err := api.NewClient(base, api.WithHTTPClient(client))
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
