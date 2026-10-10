package modelservice

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	// HostRuntimeEndpointEnv selects the CLI-owned Apple process supervisor.
	HostRuntimeEndpointEnv = "VLLM_SR_HOST_RUNTIME_ENDPOINT"
	// HostRuntimeTokenEnv names the environment variable holding the supervisor credential.
	HostRuntimeTokenEnv = "VLLM_SR_HOST_RUNTIME_TOKEN" // #nosec G101 -- environment variable name, not a credential
)

// hostRuntime is a CLI-owned host supervisor. Its private lease API changes
// process placement only; model calls retain the generated HTTP/JSON contract.
type hostRuntime struct {
	endpoint string
	token    string
}

func configuredHostRuntime() *hostRuntime {
	endpoint := strings.TrimRight(os.Getenv(HostRuntimeEndpointEnv), "/")
	if endpoint == "" {
		return nil
	}
	return &hostRuntime{endpoint: endpoint, token: os.Getenv(HostRuntimeTokenEnv)}
}

func (m *Manager) processPlans(deployments map[string]config.ModelDeployment) []*processPlan {
	if m.host == nil {
		return planProcesses(deployments, m.command, m.cacheDir, m.cores, m.resolveAuto(deployments))
	}
	placed := make(map[string]config.ModelDeployment, len(deployments))
	for name, deployment := range deployments {
		deployment = deployment.WithDefaults()
		if deployment.Managed() {
			deployment.Device = "mps"
		}
		if len(deployment.Replicas) > 0 {
			deployment.Replicas = append([]config.ModelReplica(nil), deployment.Replicas...)
			for i := range deployment.Replicas {
				if deployment.Replicas[i].Endpoint == "" {
					deployment.Replicas[i].Device = "mps"
				}
			}
		}
		placed[name] = deployment
	}
	return planProcesses(placed, m.command, "", m.cores, "mps")
}

func (h *hostRuntime) control(ctx context.Context, plan *processPlan, method string) error {
	if h.token == "" {
		return fmt.Errorf("host model runtime requires a private supervisor credential")
	}
	key := strings.TrimPrefix(plan.key, "managed\x00")
	data, err := json.Marshal(struct {
		Models []modelEntry `json:"models"`
	}{plan.models})
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, method, h.endpoint+"/processes/"+key, bytes.NewReader(data))
	if err != nil {
		return err
	}
	request.Header.Set("Authorization", "Bearer "+h.token)
	request.Header.Set("Content-Type", "application/json")
	_, client, err := newHTTPClient(h.endpoint)
	if err != nil {
		return err
	}
	client.Transport.(*http.Transport).Proxy = nil // host bridge traffic never uses an external proxy
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	response, err := client.Do(request)
	if err != nil {
		return fmt.Errorf("host runtime supervisor is unreachable: %w", err)
	}
	defer response.Body.Close()
	defer client.CloseIdleConnections()
	body, err := io.ReadAll(io.LimitReader(response.Body, 4096))
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("host runtime supervisor returned %d: %s", response.StatusCode, body)
	}
	return nil
}

func (h *hostRuntime) start(plan *processPlan) (*group, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := h.control(ctx, plan, http.MethodPost); err != nil {
		return nil, err
	}
	key := strings.TrimPrefix(plan.key, "managed\x00")
	endpoint := h.endpoint + "/processes/" + key
	base, httpClient, err := newHTTPClient(endpoint)
	if err != nil {
		return nil, err
	}
	httpClient.Transport.(*http.Transport).Proxy = nil
	httpClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	generated, err := api.NewClientWithResponses(base, api.WithHTTPClient(httpClient),
		api.WithRequestEditorFn(func(_ context.Context, req *http.Request) error {
			req.Header.Set("Authorization", "Bearer "+h.token)
			return nil
		}))
	if err != nil {
		return nil, err
	}
	client := &Client{endpoint: endpoint, api: generated}
	client.bundleTasks.Store(DefaultBundleTasks)
	g := newGroup(plan, client, true)
	g.host = h
	g.start()
	return g, nil
}

// keepAlive renews the process lease independently of inference calls. If the
// container dies without releasing it, the host reaps the process in 45 seconds.
func (h *hostRuntime) keepAlive(ctx context.Context, plan *processPlan) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	defer func() {
		stop, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		if err := h.control(stop, plan, http.MethodDelete); err != nil {
			logging.ComponentWarnEvent("model_runtime", "host_process_release_failed", map[string]interface{}{"error": err.Error()})
		}
	}()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			probe, cancel := context.WithTimeout(ctx, 15*time.Second)
			err := h.control(probe, plan, http.MethodPost)
			cancel()
			if err != nil {
				logging.ComponentWarnEvent("model_runtime", "host_process_lease_failed", map[string]interface{}{"error": err.Error()})
			}
		}
	}
}
