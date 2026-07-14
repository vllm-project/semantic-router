package aigateway

import (
	"context"
	"errors"
	"fmt"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/ai-gateway/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
}

// Profile implements the default Kubernetes baseline test profile.
type Profile struct {
	stack                 gatewayStack
	resolveGatewayService gatewayServiceResolver
	servicesForNamespace  looperGatewayServicesFactory
}

type gatewayStack interface {
	Setup(context.Context, *framework.SetupOptions) error
	Teardown(context.Context, *framework.TeardownOptions) error
	ServiceConfig() framework.ServiceConfig
}

type gatewayServiceResolver func(
	context.Context,
	*kubernetes.Clientset,
	string,
	string,
	bool,
) (string, error)

type looperGatewayServicesFactory func(
	*kubernetes.Clientset,
	string,
) looperGatewayServices

// NewProfile creates the default Kubernetes profile backed by the shared AI Gateway stack.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "ai-gateway",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
		}),
		resolveGatewayService: helpers.GetServiceByLabelInNamespace,
		servicesForNamespace: func(client *kubernetes.Clientset, namespace string) looperGatewayServices {
			return client.CoreV1().Services(namespace)
		},
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "kubernetes"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests Semantic Router through the default Kubernetes baseline powered by Envoy AI Gateway"
}

// Setup deploys the shared gateway stack.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) (setupErr error) {
	if err := p.stack.Setup(ctx, opts); err != nil {
		return err
	}
	defer func() {
		if setupErr == nil {
			return
		}
		cleanupErr := p.Teardown(
			context.WithoutCancel(ctx),
			teardownOptionsFromSetup(opts),
		)
		setupErr = errors.Join(setupErr, cleanupErr)
	}()

	if opts.KubeClient == nil {
		return fmt.Errorf("kube client is required for Looper gateway alias setup")
	}

	gatewayExternalName, err := p.looperGatewayExternalName(ctx, opts.KubeClient, opts.Verbose)
	if err != nil {
		return err
	}
	return ensureLooperGatewayAlias(
		ctx,
		p.servicesForNamespace(opts.KubeClient, helm.SemanticRouterRelease.Namespace),
		helm.SemanticRouterRelease.Namespace,
		gatewayExternalName,
	)
}

// Teardown removes the shared gateway stack.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	var aliasErr error
	if opts.KubeClient != nil {
		aliasErr = deleteOwnedLooperGatewayAlias(
			ctx,
			p.servicesForNamespace(opts.KubeClient, helm.SemanticRouterRelease.Namespace),
			helm.SemanticRouterRelease.Namespace,
			func() (string, error) {
				return p.looperGatewayExternalName(ctx, opts.KubeClient, opts.Verbose)
			},
		)
	}
	return errors.Join(aliasErr, p.stack.Teardown(ctx, opts))
}

func (p *Profile) looperGatewayExternalName(
	ctx context.Context,
	client *kubernetes.Clientset,
	verbose bool,
) (string, error) {
	serviceConfig := p.stack.ServiceConfig()
	gatewayServiceName, err := p.resolveGatewayService(
		ctx,
		client,
		serviceConfig.Namespace,
		serviceConfig.LabelSelector,
		verbose,
	)
	if err != nil {
		return "", fmt.Errorf("resolve Looper gateway service: %w", err)
	}
	return fmt.Sprintf(
		"%s.%s.svc.cluster.local",
		gatewayServiceName,
		serviceConfig.Namespace,
	), nil
}

func teardownOptionsFromSetup(opts *framework.SetupOptions) *framework.TeardownOptions {
	return &framework.TeardownOptions{
		KubeClient:  opts.KubeClient,
		KubeConfig:  opts.KubeConfig,
		ClusterName: opts.ClusterName,
		Verbose:     opts.Verbose,
	}
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return testmatrix.Combine(testmatrix.BaselineRouterContract)
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
