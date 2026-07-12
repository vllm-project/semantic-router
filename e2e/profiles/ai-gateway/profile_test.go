package aigateway

import (
	"context"
	"errors"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

type memoryLooperGatewayServices struct {
	service     *corev1.Service
	createErr   error
	deleteErr   error
	deleteCalls int
}

func (m *memoryLooperGatewayServices) Get(
	_ context.Context,
	name string,
	_ metav1.GetOptions,
) (*corev1.Service, error) {
	if m.service == nil || m.service.Name != name {
		return nil, apierrors.NewNotFound(schema.GroupResource{Resource: "services"}, name)
	}
	return m.service.DeepCopy(), nil
}

func (m *memoryLooperGatewayServices) Create(
	_ context.Context,
	service *corev1.Service,
	_ metav1.CreateOptions,
) (*corev1.Service, error) {
	if m.createErr != nil {
		return nil, m.createErr
	}
	if m.service != nil {
		return nil, apierrors.NewAlreadyExists(
			schema.GroupResource{Resource: "services"},
			service.Name,
		)
	}
	m.service = service.DeepCopy()
	return m.service.DeepCopy(), nil
}

func (m *memoryLooperGatewayServices) Delete(
	_ context.Context,
	name string,
	_ metav1.DeleteOptions,
) error {
	m.deleteCalls++
	if m.deleteErr != nil {
		return m.deleteErr
	}
	if m.service == nil || m.service.Name != name {
		return apierrors.NewNotFound(schema.GroupResource{Resource: "services"}, name)
	}
	m.service = nil
	return nil
}

type memoryGatewayStack struct {
	serviceConfig framework.ServiceConfig
	setupErr      error
	teardownErr   error
	setupCalls    int
	teardownCalls int
	teardownOpts  *framework.TeardownOptions
}

func (m *memoryGatewayStack) Setup(context.Context, *framework.SetupOptions) error {
	m.setupCalls++
	return m.setupErr
}

func (m *memoryGatewayStack) Teardown(
	_ context.Context,
	opts *framework.TeardownOptions,
) error {
	m.teardownCalls++
	m.teardownOpts = opts
	return m.teardownErr
}

func (m *memoryGatewayStack) ServiceConfig() framework.ServiceConfig {
	return m.serviceConfig
}

func TestProfileRegistersLooperSecurityContract(t *testing.T) {
	const testcaseName = "looper-secret-forgery"
	registered, err := pkgtestcases.ListByNames(testcaseName)
	if err != nil || len(registered) != 1 || registered[0].Fn == nil {
		t.Fatalf("registered Looper testcase = %#v, err = %v", registered, err)
	}

	found := false
	for _, name := range NewProfile().GetTestCases() {
		if name == testcaseName {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("baseline profile does not include registered testcase %q", testcaseName)
	}
}

func TestEnsureLooperGatewayAliasCreatesStableExternalName(t *testing.T) {
	services := &memoryLooperGatewayServices{}
	const (
		namespace    = "router-system"
		externalName = "generated-gateway.envoy-system.svc.cluster.local"
	)

	if err := ensureLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		externalName,
	); err != nil {
		t.Fatalf("ensureLooperGatewayAlias() error = %v", err)
	}

	service := services.service
	if service.Spec.Type != corev1.ServiceTypeExternalName {
		t.Fatalf("alias service type = %q, want ExternalName", service.Spec.Type)
	}
	if service.Spec.ExternalName != externalName {
		t.Fatalf("alias target = %q, want %q", service.Spec.ExternalName, externalName)
	}
	if len(service.Spec.Ports) != 1 || service.Spec.Ports[0].Port != 80 {
		t.Fatalf("alias ports = %#v, want one HTTP port 80", service.Spec.Ports)
	}
	if !isOwnedLooperGatewayAlias(service) {
		t.Fatalf("created alias labels = %#v, want E2E ownership identity", service.Labels)
	}

	if err := ensureLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		externalName,
	); err != nil {
		t.Fatalf("idempotent ensureLooperGatewayAlias() error = %v", err)
	}
}

func TestEnsureLooperGatewayAliasRejectsConflictingService(t *testing.T) {
	services := &memoryLooperGatewayServices{service: &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      looperGatewayAliasName,
			Namespace: "router-system",
		},
		Spec: corev1.ServiceSpec{
			Type:         corev1.ServiceTypeExternalName,
			ExternalName: "unexpected.example.test",
		},
	}}

	err := ensureLooperGatewayAlias(
		context.Background(),
		services,
		"router-system",
		"generated-gateway.envoy-system.svc.cluster.local",
	)
	if err == nil || !strings.Contains(err.Error(), "expected ownership and specification") {
		t.Fatalf("conflicting alias error = %v, want ownership/specification conflict", err)
	}
}

func TestEnsureLooperGatewayAliasRejectsUnownedMatchingService(t *testing.T) {
	const (
		namespace    = "router-system"
		externalName = "generated-gateway.envoy-system.svc.cluster.local"
	)
	services := &memoryLooperGatewayServices{service: looperGatewayAliasFixture(
		namespace,
		externalName,
		false,
	)}

	err := ensureLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		externalName,
	)
	if err == nil || !strings.Contains(err.Error(), "expected ownership") {
		t.Fatalf("unowned matching alias error = %v, want ownership rejection", err)
	}
	if services.deleteCalls != 0 {
		t.Fatalf("unowned matching alias delete calls = %d, want 0", services.deleteCalls)
	}
}

func TestEnsureLooperGatewayAliasRejectsOwnedServiceWithWrongPort(t *testing.T) {
	const (
		namespace    = "router-system"
		externalName = "generated-gateway.envoy-system.svc.cluster.local"
	)
	service := looperGatewayAliasFixture(namespace, externalName, true)
	service.Spec.Ports[0].Port = 8080
	services := &memoryLooperGatewayServices{service: service}

	err := ensureLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		externalName,
	)
	if err == nil || !strings.Contains(err.Error(), "expected ownership and specification") {
		t.Fatalf("wrong-port alias error = %v, want specification rejection", err)
	}
}

func TestDeleteOwnedLooperGatewayAliasRefusesUnownedService(t *testing.T) {
	const namespace = "router-system"
	services := &memoryLooperGatewayServices{service: looperGatewayAliasFixture(
		namespace,
		"generated-gateway.envoy-system.svc.cluster.local",
		false,
	)}

	err := deleteOwnedLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		fixedLooperGatewayExternalName("generated-gateway.envoy-system.svc.cluster.local"),
	)
	if err == nil || !strings.Contains(err.Error(), "refusing to delete unowned") {
		t.Fatalf("unowned alias deletion error = %v, want refusal", err)
	}
	if services.deleteCalls != 0 {
		t.Fatalf("unowned alias delete calls = %d, want 0", services.deleteCalls)
	}
	if services.service == nil {
		t.Fatal("unowned alias was deleted")
	}
}

func TestDeleteOwnedLooperGatewayAliasIsIdempotent(t *testing.T) {
	const namespace = "router-system"
	services := &memoryLooperGatewayServices{service: looperGatewayAliasFixture(
		namespace,
		"generated-gateway.envoy-system.svc.cluster.local",
		true,
	)}

	expectedExternalName := fixedLooperGatewayExternalName(
		"generated-gateway.envoy-system.svc.cluster.local",
	)
	if err := deleteOwnedLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		expectedExternalName,
	); err != nil {
		t.Fatalf("deleteOwnedLooperGatewayAlias() error = %v", err)
	}
	if err := deleteOwnedLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		expectedExternalName,
	); err != nil {
		t.Fatalf("idempotent deleteOwnedLooperGatewayAlias() error = %v", err)
	}
	if services.deleteCalls != 1 {
		t.Fatalf("owned alias delete calls = %d, want 1", services.deleteCalls)
	}
}

func TestDeleteOwnedLooperGatewayAliasRefusesMismatchedService(t *testing.T) {
	const namespace = "router-system"
	services := &memoryLooperGatewayServices{service: looperGatewayAliasFixture(
		namespace,
		"unexpected.envoy-system.svc.cluster.local",
		true,
	)}

	err := deleteOwnedLooperGatewayAlias(
		context.Background(),
		services,
		namespace,
		fixedLooperGatewayExternalName("generated-gateway.envoy-system.svc.cluster.local"),
	)
	if err == nil || !strings.Contains(err.Error(), "refusing to delete mismatched") {
		t.Fatalf("mismatched alias deletion error = %v, want refusal", err)
	}
	if services.deleteCalls != 0 || services.service == nil {
		t.Fatalf(
			"mismatched alias state = delete calls:%d service:%v, want preserved",
			services.deleteCalls,
			services.service,
		)
	}
}

func TestProfileTeardownRefusesUnownedAliasAndStillCleansStack(t *testing.T) {
	const namespace = "router-system"
	stackCleanupErr := errors.New("stack cleanup failed")
	stack := &memoryGatewayStack{teardownErr: stackCleanupErr}
	services := &memoryLooperGatewayServices{service: looperGatewayAliasFixture(
		namespace,
		"generated-gateway.envoy-system.svc.cluster.local",
		false,
	)}
	profile := &Profile{
		stack: stack,
		servicesForNamespace: func(*kubernetes.Clientset, string) looperGatewayServices {
			return services
		},
	}

	err := profile.Teardown(context.Background(), &framework.TeardownOptions{
		KubeClient: &kubernetes.Clientset{},
	})
	if err == nil || !strings.Contains(err.Error(), "refusing to delete unowned") ||
		!errors.Is(err, stackCleanupErr) {
		t.Fatalf("Teardown() error = %v, want joined ownership refusal and stack error", err)
	}
	if services.deleteCalls != 0 || services.service == nil {
		t.Fatalf(
			"unowned alias state = delete calls:%d service:%v, want preserved",
			services.deleteCalls,
			services.service,
		)
	}
	if stack.teardownCalls != 1 {
		t.Fatalf("stack teardown calls = %d, want 1", stack.teardownCalls)
	}
}

func TestProfileSetupRollsBackStackWhenAliasResolutionFails(t *testing.T) {
	resolveErr := errors.New("gateway lookup failed")
	cleanupErr := errors.New("stack cleanup failed")
	stack := &memoryGatewayStack{
		serviceConfig: framework.ServiceConfig{
			Namespace:     "envoy-system",
			LabelSelector: "gateway=semantic-router",
		},
		teardownErr: cleanupErr,
	}
	services := &memoryLooperGatewayServices{}
	profile := &Profile{
		stack: stack,
		resolveGatewayService: func(
			context.Context,
			*kubernetes.Clientset,
			string,
			string,
			bool,
		) (string, error) {
			return "", resolveErr
		},
		servicesForNamespace: func(*kubernetes.Clientset, string) looperGatewayServices {
			return services
		},
	}
	opts := &framework.SetupOptions{
		KubeClient:  &kubernetes.Clientset{},
		KubeConfig:  "/tmp/kubeconfig",
		ClusterName: "rollback-cluster",
		Verbose:     true,
	}

	err := profile.Setup(context.Background(), opts)
	if !errors.Is(err, resolveErr) || !errors.Is(err, cleanupErr) {
		t.Fatalf("Setup() error = %v, want joined resolution and cleanup errors", err)
	}
	if stack.setupCalls != 1 || stack.teardownCalls != 1 {
		t.Fatalf(
			"stack calls = setup:%d teardown:%d, want 1 each",
			stack.setupCalls,
			stack.teardownCalls,
		)
	}
	if stack.teardownOpts == nil ||
		stack.teardownOpts.KubeClient != opts.KubeClient ||
		stack.teardownOpts.KubeConfig != opts.KubeConfig ||
		stack.teardownOpts.ClusterName != opts.ClusterName ||
		stack.teardownOpts.Verbose != opts.Verbose {
		t.Fatalf("rollback teardown options = %#v, want setup identity fields", stack.teardownOpts)
	}
}

func TestProfileSetupRollsBackStackWhenAliasCreateFails(t *testing.T) {
	createErr := errors.New("service create failed")
	stack := &memoryGatewayStack{serviceConfig: framework.ServiceConfig{
		Namespace:     "envoy-system",
		LabelSelector: "gateway=semantic-router",
	}}
	services := &memoryLooperGatewayServices{createErr: createErr}
	profile := &Profile{
		stack: stack,
		resolveGatewayService: func(
			context.Context,
			*kubernetes.Clientset,
			string,
			string,
			bool,
		) (string, error) {
			return "generated-gateway", nil
		},
		servicesForNamespace: func(*kubernetes.Clientset, string) looperGatewayServices {
			return services
		},
	}

	err := profile.Setup(context.Background(), &framework.SetupOptions{
		KubeClient: &kubernetes.Clientset{},
	})
	if !errors.Is(err, createErr) {
		t.Fatalf("Setup() error = %v, want alias create failure", err)
	}
	if stack.teardownCalls != 1 {
		t.Fatalf("stack teardown calls = %d, want 1", stack.teardownCalls)
	}
}

func looperGatewayAliasFixture(
	namespace string,
	externalName string,
	owned bool,
) *corev1.Service {
	labels := map[string]string{}
	if owned {
		labels[looperGatewayAliasManagedByLabel] = looperGatewayAliasManagedBy
		labels[looperGatewayAliasComponentLabel] = looperGatewayAliasComponent
	}
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      looperGatewayAliasName,
			Namespace: namespace,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Type:         corev1.ServiceTypeExternalName,
			ExternalName: externalName,
			Ports: []corev1.ServicePort{{
				Name:     "http",
				Protocol: corev1.ProtocolTCP,
				Port:     80,
			}},
		},
	}
}

func fixedLooperGatewayExternalName(externalName string) func() (string, error) {
	return func() (string, error) {
		return externalName, nil
	}
}
