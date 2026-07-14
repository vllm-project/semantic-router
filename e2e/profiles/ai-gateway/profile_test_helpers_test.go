package aigateway

import (
	"context"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
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
