package dashboard

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
)

const (
	routerBootstrapConfigVolume        = "config-volume"
	dashboardRouterConfigVolume        = "router-config"
	dashboardRouterConfigMapUnresolved = "router-config-resolved-at-setup"
)

func (p *Profile) applyDashboardDeployment(ctx context.Context, opts *framework.SetupOptions) error {
	if opts == nil || opts.KubeClient == nil {
		return fmt.Errorf("Kubernetes client is required to deploy Dashboard")
	}

	router, err := opts.KubeClient.AppsV1().Deployments(namespaceRouter).Get(
		ctx,
		"semantic-router",
		metav1.GetOptions{},
	)
	if err != nil {
		return fmt.Errorf("read Router deployment: %w", err)
	}
	configMapName, err := routerBootstrapConfigMapName(router)
	if err != nil {
		return err
	}
	configMap, err := opts.KubeClient.CoreV1().ConfigMaps(namespaceRouter).Get(
		ctx,
		configMapName,
		metav1.GetOptions{},
	)
	if err != nil {
		return fmt.Errorf("read Router bootstrap ConfigMap %q: %w", configMapName, err)
	}
	if configMap.Immutable == nil || !*configMap.Immutable {
		return fmt.Errorf("Router bootstrap ConfigMap %q must be immutable", configMapName)
	}

	desired, err := loadDashboardDeployment(dashboardE2EDeploymentManifest, configMapName)
	if err != nil {
		return err
	}
	if desired.Labels == nil {
		desired.Labels = make(map[string]string)
	}
	for key, value := range e2eManagedLabels {
		desired.Labels[key] = value
	}

	deployments := opts.KubeClient.AppsV1().Deployments(namespaceRouter)
	existing, err := deployments.Get(ctx, desired.Name, metav1.GetOptions{})
	switch {
	case apierrors.IsNotFound(err):
		if _, createErr := deployments.Create(ctx, desired, metav1.CreateOptions{}); createErr != nil {
			return fmt.Errorf("create Dashboard deployment: %w", createErr)
		}
		return nil
	case err != nil:
		return fmt.Errorf("read existing Dashboard deployment: %w", err)
	case existing.Labels["vllm.ai/e2e-managed"] != "true":
		return fmt.Errorf("refuse to replace unmanaged Dashboard deployment %s/%s", namespaceRouter, desired.Name)
	default:
		desired.ResourceVersion = existing.ResourceVersion
		if _, updateErr := deployments.Update(ctx, desired, metav1.UpdateOptions{}); updateErr != nil {
			return fmt.Errorf("update Dashboard deployment: %w", updateErr)
		}
		return nil
	}
}

func routerBootstrapConfigMapName(deployment *appsv1.Deployment) (string, error) {
	if deployment == nil {
		return "", fmt.Errorf("Router deployment is required")
	}
	name := ""
	for _, volume := range deployment.Spec.Template.Spec.Volumes {
		if volume.Name != routerBootstrapConfigVolume {
			continue
		}
		if name != "" {
			return "", fmt.Errorf("Router deployment has more than one %q volume", routerBootstrapConfigVolume)
		}
		if volume.ConfigMap == nil {
			return "", fmt.Errorf("Router bootstrap volume %q is not a ConfigMap source", routerBootstrapConfigVolume)
		}
		name = strings.TrimSpace(volume.ConfigMap.Name)
		if name == "" {
			return "", fmt.Errorf("Router bootstrap volume %q has no ConfigMap name", routerBootstrapConfigVolume)
		}
	}
	if name == "" {
		return "", fmt.Errorf("Router deployment has no %q ConfigMap volume", routerBootstrapConfigVolume)
	}
	return name, nil
}

func loadDashboardDeployment(path string, routerConfigMapName string) (*appsv1.Deployment, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read Dashboard deployment manifest: %w", err)
	}
	document, err := utilyaml.ToJSON(raw)
	if err != nil {
		return nil, fmt.Errorf("decode Dashboard deployment manifest: %w", err)
	}
	var deployment appsv1.Deployment
	if err := json.Unmarshal(document, &deployment); err != nil {
		return nil, fmt.Errorf("decode Dashboard deployment: %w", err)
	}
	if deployment.Name != deploymentDashboard {
		return nil, fmt.Errorf("Dashboard deployment manifest names %q, want %q", deployment.Name, deploymentDashboard)
	}
	deployment.Namespace = namespaceRouter
	if err := bindDashboardRouterConfig(&deployment, routerConfigMapName); err != nil {
		return nil, err
	}
	return &deployment, nil
}

func bindDashboardRouterConfig(deployment *appsv1.Deployment, configMapName string) error {
	if deployment == nil {
		return fmt.Errorf("Dashboard deployment is required")
	}
	configMapName = strings.TrimSpace(configMapName)
	if configMapName == "" {
		return fmt.Errorf("Router bootstrap ConfigMap name is required")
	}
	boundIndex := -1
	for index := range deployment.Spec.Template.Spec.Volumes {
		volume := &deployment.Spec.Template.Spec.Volumes[index]
		if volume.Name != dashboardRouterConfigVolume {
			continue
		}
		if boundIndex >= 0 {
			return fmt.Errorf("Dashboard deployment has more than one %q volume", dashboardRouterConfigVolume)
		}
		if volume.ConfigMap == nil {
			return fmt.Errorf("Dashboard Router config volume %q is not a ConfigMap source", dashboardRouterConfigVolume)
		}
		if volume.ConfigMap.Name != dashboardRouterConfigMapUnresolved {
			return fmt.Errorf(
				"Dashboard Router config volume must use the setup placeholder %q",
				dashboardRouterConfigMapUnresolved,
			)
		}
		boundIndex = index
	}
	if boundIndex < 0 {
		return fmt.Errorf("Dashboard deployment has no %q ConfigMap volume", dashboardRouterConfigVolume)
	}
	deployment.Spec.Template.Spec.Volumes[boundIndex].ConfigMap.Name = configMapName
	return nil
}
