package testcases

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// Names from e2e/profiles/model-runtime/values.yaml, which its profile test
// checks against this file.
const (
	mrSocketDir           = "/tmp/vsr-runtime"
	mrDecisionDeployment  = "decision-fixture"
	mrDomainDeployment    = "vela-domain"
	mrPIIDeployment       = "vela-pii"
	mrGuardDeployment     = "vela-guard"
	mrEmbeddingDeployment = "vela-embedding"
	mrRerankerDeployment  = "vela-reranker"
	mrModalityDeployment  = "vela-modality"
	mrAttachedDecisions   = "attached-decisions"
	mrAttachedFeedback    = "attached-feedback"
	mrAttachedVela2       = "attached-vela2"
	mrOfflineDeployment   = "decision-offline"
	mrAttachedService     = "model-runtime-attached"
	mrReadyTimeout        = 5 * time.Minute
	mrRequestTimeout      = 60 * time.Second
)

var (
	// Sorted, as the lifecycle case compares it with each process's models.
	mrDeviceGroup         = []string{mrDomainDeployment, mrEmbeddingDeployment, mrGuardDeployment, mrModalityDeployment, mrPIIDeployment, mrRerankerDeployment}
	mrManagedDeployments  = append([]string{mrDecisionDeployment}, mrDeviceGroup...)
	mrAttachedDeployments = []string{mrAttachedDecisions, mrAttachedFeedback, mrAttachedVela2}
)

// modelRuntimeSession holds the connections the model-runtime contracts use.
type modelRuntimeSession struct {
	gatewayPort string
	api         *fixtures.ServiceSession
	metrics     *fixtures.ServiceSession
	attached    *fixtures.ServiceSession
	pod         modelruntime.PodTarget
	closers     []func()
}

// openModelRuntimeSession opens the Router's connections and the model-runtime
// profile's attached runtime.
func openModelRuntimeSession(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*modelRuntimeSession, error) {
	session, err := openRouterRuntimeSession(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	session.attached, err = fixtures.OpenServiceEndpointSession(ctx, client, opts, modelruntime.RouterNamespace, mrAttachedService, "8100")
	if err != nil {
		session.Close()
		return nil, err
	}
	session.closers = append(session.closers, session.attached.Close)
	return session, nil
}

// openRouterRuntimeSession opens the gateway, Router API and metrics
// connections and finds the Router pod that runs the managed runtimes.
func openRouterRuntimeSession(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*modelRuntimeSession, error) {
	session := &modelRuntimeSession{}
	gatewayPort, stopGateway, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	session.gatewayPort = gatewayPort
	session.closers = append(session.closers, stopGateway)
	if session.api, err = fixtures.OpenRouterAPISession(ctx, client, opts); err != nil {
		session.Close()
		return nil, err
	}
	session.closers = append(session.closers, session.api.Close)
	if session.metrics, err = fixtures.OpenSemanticRouterMetricsSession(ctx, client, opts); err != nil {
		session.Close()
		return nil, err
	}
	session.closers = append(session.closers, session.metrics.Close)
	if session.pod, err = modelruntime.RouterPod(ctx, client, opts.RestConfig); err != nil {
		session.Close()
		return nil, err
	}
	return session, nil
}

func (s *modelRuntimeSession) Close() {
	for index := len(s.closers) - 1; index >= 0; index-- {
		s.closers[index]()
	}
}

// routerMetrics scrapes the Router's Prometheus endpoint.
func (s *modelRuntimeSession) routerMetrics(ctx context.Context) (modelruntime.Metrics, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, s.metrics.URL("/metrics"), nil)
	if err != nil {
		return modelruntime.Metrics{}, err
	}
	response, err := s.metrics.HTTPClient(30 * time.Second).Do(request)
	if err != nil {
		return modelruntime.Metrics{}, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return modelruntime.Metrics{}, err
	}
	if response.StatusCode != http.StatusOK {
		return modelruntime.Metrics{}, fmt.Errorf("router metrics returned %d", response.StatusCode)
	}
	return modelruntime.ParseMetrics(string(body))
}

// waitReady waits until the Router reports every named deployment ready.
func (s *modelRuntimeSession) waitReady(ctx context.Context, deployments ...string) error {
	return s.waitReadyWithin(ctx, mrReadyTimeout, deployments...)
}

// waitReadyWithin is waitReady with its own bound, for models that download first.
func (s *modelRuntimeSession) waitReadyWithin(ctx context.Context, timeout time.Duration, deployments ...string) error {
	return modelruntime.Eventually(ctx, timeout, func(ctx context.Context) error {
		metrics, err := s.routerMetrics(ctx)
		if err != nil {
			return err
		}
		var waiting []string
		for _, name := range deployments {
			if !metrics.DeploymentReady(name) {
				waiting = append(waiting, name)
			}
		}
		if len(waiting) > 0 {
			return fmt.Errorf("deployments not ready: %v", waiting)
		}
		return nil
	})
}

// managed returns a client for the managed runtime process serving deployment.
func (s *modelRuntimeSession) managed(ctx context.Context, deployment string) (*modelruntime.Client, string, error) {
	return s.pod.ClientFor(ctx, mrSocketDir, deployment)
}

// attachedRuntime returns a client for the attached runtime.
func (s *modelRuntimeSession) attachedRuntime() *modelruntime.Client {
	return modelruntime.NewHTTPClient(s.attached.BaseURL(), mrRequestTimeout)
}

// routingPreview is the subset of /api/v1/routing/preview the contracts read.
type routingPreview struct {
	OriginalText      string             `json:"original_text"`
	SignalConfidences map[string]float64 `json:"signal_confidences"`
	SignalValues      map[string]float64 `json:"signal_values"`
	SignalErrors      map[string]string  `json:"signal_errors"`
	DecisionResult    *struct {
		DecisionName string `json:"decision_name"`
	} `json:"decision_result"`
}

// preview evaluates one user message without generating an answer.
func (s *modelRuntimeSession) preview(ctx context.Context, text string) (routingPreview, error) {
	return s.previewConversation(ctx, []map[string]string{{"role": "user", "content": text}})
}

// previewConversation evaluates a whole conversation without generating an answer.
func (s *modelRuntimeSession) previewConversation(ctx context.Context, messages []map[string]string) (routingPreview, error) {
	var preview routingPreview
	err := s.postAPI(ctx, "/api/v1/routing/preview?trace=true", map[string]interface{}{
		"model":    "vllm-sr/auto",
		"messages": messages,
	}, &preview)
	return preview, err
}

// chat sends one user message through the gateway with the debug headers on.
func (s *modelRuntimeSession) chat(ctx context.Context, text string) (*localChatCompletionResponse, error) {
	response, err := sendLocalChatCompletion(ctx, s.gatewayPort, "vllm-sr/auto", text, mrRequestTimeout)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("chat %q: %s", text, formatUnexpectedChatCompletionStatus(response))
	}
	return response, nil
}

// headerItems splits a comma-separated x-vsr header into its entries.
func headerItems(headers http.Header, name string) map[string]bool {
	items := map[string]bool{}
	for _, item := range strings.Split(headers.Get(name), ",") {
		if item = strings.TrimSpace(item); item != "" {
			items[item] = true
		}
	}
	return items
}

// freshPrompt returns text no earlier request sent, so the Router's result
// cache cannot answer it and the call must reach the runtime, also on a retry.
func freshPrompt(text string) string {
	return fmt.Sprintf("%s (request %d)", text, time.Now().UnixNano())
}
