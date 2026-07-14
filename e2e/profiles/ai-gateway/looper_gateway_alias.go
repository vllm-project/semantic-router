package aigateway

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	looperGatewayAliasName           = "semantic-router-looper"
	looperGatewayAliasManagedByLabel = "app.kubernetes.io/managed-by"
	looperGatewayAliasManagedBy      = "semantic-router-e2e"
	looperGatewayAliasComponentLabel = "app.kubernetes.io/component"
	looperGatewayAliasComponent      = "looper-gateway-alias"
)

type looperGatewayServices interface {
	Get(context.Context, string, metav1.GetOptions) (*corev1.Service, error)
	Create(context.Context, *corev1.Service, metav1.CreateOptions) (*corev1.Service, error)
	Delete(context.Context, string, metav1.DeleteOptions) error
}

func ensureLooperGatewayAlias(
	ctx context.Context,
	services looperGatewayServices,
	namespace string,
	externalName string,
) error {
	existing, err := services.Get(ctx, looperGatewayAliasName, metav1.GetOptions{})
	if err == nil {
		if isOwnedLooperGatewayAlias(existing) &&
			isExpectedLooperGatewayAlias(existing, namespace, externalName) {
			return nil
		}
		return fmt.Errorf(
			"Looper gateway alias %s/%s already exists without the expected ownership and specification",
			namespace,
			looperGatewayAliasName,
		)
	}
	if !apierrors.IsNotFound(err) {
		return fmt.Errorf("inspect Looper gateway alias: %w", err)
	}

	_, err = services.Create(ctx, &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      looperGatewayAliasName,
			Namespace: namespace,
			Labels: map[string]string{
				looperGatewayAliasManagedByLabel: looperGatewayAliasManagedBy,
				looperGatewayAliasComponentLabel: looperGatewayAliasComponent,
			},
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
	}, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("create Looper gateway alias: %w", err)
	}
	return nil
}

func deleteOwnedLooperGatewayAlias(
	ctx context.Context,
	services looperGatewayServices,
	namespace string,
	expectedExternalName func() (string, error),
) error {
	existing, err := services.Get(ctx, looperGatewayAliasName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("inspect Looper gateway alias before deletion: %w", err)
	}
	if !isOwnedLooperGatewayAlias(existing) {
		return fmt.Errorf(
			"refusing to delete unowned Looper gateway alias %s/%s",
			namespace,
			looperGatewayAliasName,
		)
	}
	externalName, err := expectedExternalName()
	if err != nil {
		return fmt.Errorf("resolve expected Looper gateway alias before deletion: %w", err)
	}
	if !isExpectedLooperGatewayAlias(existing, namespace, externalName) {
		return fmt.Errorf(
			"refusing to delete mismatched Looper gateway alias %s/%s",
			namespace,
			looperGatewayAliasName,
		)
	}
	if err := services.Delete(ctx, looperGatewayAliasName, metav1.DeleteOptions{}); err != nil &&
		!apierrors.IsNotFound(err) {
		return fmt.Errorf("delete Looper gateway alias: %w", err)
	}
	return nil
}

func isOwnedLooperGatewayAlias(service *corev1.Service) bool {
	return service != nil &&
		service.Labels[looperGatewayAliasManagedByLabel] == looperGatewayAliasManagedBy &&
		service.Labels[looperGatewayAliasComponentLabel] == looperGatewayAliasComponent
}

func isExpectedLooperGatewayAlias(
	service *corev1.Service,
	namespace string,
	externalName string,
) bool {
	if service == nil || service.Name != looperGatewayAliasName ||
		service.Namespace != namespace ||
		service.Spec.Type != corev1.ServiceTypeExternalName ||
		service.Spec.ExternalName != externalName ||
		len(service.Spec.Ports) != 1 {
		return false
	}
	port := service.Spec.Ports[0]
	return port.Name == "http" && port.Protocol == corev1.ProtocolTCP && port.Port == 80
}
