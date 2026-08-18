package dashboard

import (
	"encoding/json"
	"os"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
)

func TestDashboardE2EStagesWritableRuntimeConfig(t *testing.T) {
	raw, err := os.ReadFile("dashboard-deployment.yaml")
	if err != nil {
		t.Fatalf("read dashboard deployment: %v", err)
	}
	jsonDocument, err := utilyaml.ToJSON(raw)
	if err != nil {
		t.Fatalf("convert dashboard deployment to JSON: %v", err)
	}
	var deployment appsv1.Deployment
	if err := json.Unmarshal(jsonDocument, &deployment); err != nil {
		t.Fatalf("decode dashboard deployment: %v", err)
	}

	initContainer := requireContainer(t, deployment.Spec.Template.Spec.InitContainers, "stage-dashboard-runtime-config")
	requireVolumeMount(t, initContainer.VolumeMounts, "router-config-seed", true)
	requireVolumeMount(t, initContainer.VolumeMounts, "router-config-runtime", false)

	dashboard := requireContainer(t, deployment.Spec.Template.Spec.Containers, "dashboard")
	requireVolumeMount(t, dashboard.VolumeMounts, "router-config-runtime", false)
	if value := envValue(dashboard.Env, "DASHBOARD_RUNTIME_CONFIG_WRITABLE"); value != "true" {
		t.Fatalf("DASHBOARD_RUNTIME_CONFIG_WRITABLE = %q, want true", value)
	}

	for _, volume := range deployment.Spec.Template.Spec.Volumes {
		if volume.Name == "router-config-runtime" && volume.EmptyDir != nil {
			return
		}
	}
	t.Fatal("router-config-runtime must be an emptyDir")
}

func requireContainer(t *testing.T, containers []corev1.Container, name string) corev1.Container {
	t.Helper()
	for _, container := range containers {
		if container.Name == name {
			return container
		}
	}
	t.Fatalf("container %q not found", name)
	return corev1.Container{}
}

func requireVolumeMount(t *testing.T, mounts []corev1.VolumeMount, name string, readOnly bool) {
	t.Helper()
	for _, mount := range mounts {
		if mount.Name != name {
			continue
		}
		if mount.ReadOnly != readOnly {
			t.Fatalf("volume mount %q readOnly = %t, want %t", name, mount.ReadOnly, readOnly)
		}
		return
	}
	t.Fatalf("volume mount %q not found", name)
}

func envValue(environment []corev1.EnvVar, name string) string {
	for _, variable := range environment {
		if variable.Name == name {
			return variable.Value
		}
	}
	return ""
}
