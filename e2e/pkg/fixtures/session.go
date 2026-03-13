package fixtures

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

// ServiceSession owns a port-forwarded local endpoint for an in-cluster service.
type ServiceSession struct {
	baseURL   string
	localPort string
	stop      func()
	closeOnce sync.Once
}

// OpenServiceSession establishes a port-forward to the profile service described by opts.
func OpenServiceSession(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (*ServiceSession, error) {
	svcConfig := opts.ServiceConfig
	if svcConfig.LabelSelector == "" && svcConfig.Name == "" {
		return nil, fmt.Errorf("service configuration is required: either LabelSelector or Name must be provided")
	}

	serviceName, err := resolveServiceName(ctx, client, svcConfig, opts.Verbose)
	if err != nil {
		return nil, err
	}
	localPort, servicePort, err := resolvePorts(svcConfig)
	if err != nil {
		return nil, err
	}

	stop, err := helpers.StartPortForward(
		ctx,
		client,
		opts.RestConfig,
		svcConfig.Namespace,
		serviceName,
		fmt.Sprintf("%s:%s", localPort, servicePort),
		opts.Verbose,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to start port forwarding: %w", err)
	}

	time.Sleep(2 * time.Second)
	return newSession(localPort, stop), nil
}

// OpenRouterAPISession establishes a port-forward to the semantic-router API service.
func OpenRouterAPISession(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (*ServiceSession, error) {
	localPort, err := getAvailablePort()
	if err != nil {
		return nil, fmt.Errorf("failed to get available port: %w", err)
	}

	stop, err := helpers.StartPortForward(
		ctx,
		client,
		opts.RestConfig,
		"vllm-semantic-router-system",
		"semantic-router",
		fmt.Sprintf("%s:8080", localPort),
		opts.Verbose,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to start router API port forwarding: %w", err)
	}

	time.Sleep(2 * time.Second)
	return newSession(localPort, stop), nil
}

// BaseURL returns the local HTTP base URL.
func (s *ServiceSession) BaseURL() string {
	return s.baseURL
}

// LocalPort returns the local port allocated for the session.
func (s *ServiceSession) LocalPort() string {
	return s.localPort
}

// URL resolves a path against the session base URL.
func (s *ServiceSession) URL(path string) string {
	return s.baseURL + path
}

// HTTPClient returns a timeout-configured HTTP client for this session.
func (s *ServiceSession) HTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{Timeout: timeout}
}

// Close stops the underlying port-forward.
func (s *ServiceSession) Close() {
	if s == nil || s.stop == nil {
		return
	}
	s.closeOnce.Do(s.stop)
}

func newSession(localPort string, stop func()) *ServiceSession {
	return &ServiceSession{
		baseURL:   fmt.Sprintf("http://localhost:%s", localPort),
		localPort: localPort,
		stop:      stop,
	}
}

func resolveServiceName(
	ctx context.Context,
	client *kubernetes.Clientset,
	svcConfig pkgtestcases.ServiceConfig,
	verbose bool,
) (string, error) {
	if svcConfig.Name != "" {
		return svcConfig.Name, nil
	}
	serviceName, err := helpers.GetServiceByLabelInNamespace(ctx, client, svcConfig.Namespace, svcConfig.LabelSelector, verbose)
	if err != nil {
		return "", fmt.Errorf("failed to get service by label selector: %w", err)
	}
	return serviceName, nil
}

func resolvePorts(svcConfig pkgtestcases.ServiceConfig) (string, string, error) {
	servicePort := svcConfig.ServicePort
	if servicePort == "" {
		_, parsedServicePort, err := helpers.ParsePortMapping(svcConfig.PortMapping)
		if err != nil {
			return "", "", fmt.Errorf("service port is required: %w", err)
		}
		servicePort = parsedServicePort
	}

	localPort := svcConfig.LocalPort
	if localPort == "" {
		var err error
		localPort, err = getAvailablePort()
		if err != nil {
			return "", "", fmt.Errorf("failed to allocate local port: %w", err)
		}
	}

	return localPort, servicePort, nil
}

func getAvailablePort() (string, error) {
	listener, err := net.Listen("tcp", ":0")
	if err != nil {
		return "", fmt.Errorf("failed to find available port: %w", err)
	}
	defer func() {
		_ = listener.Close()
	}()
	addr := listener.Addr().(*net.TCPAddr)
	return fmt.Sprintf("%d", addr.Port), nil
}
