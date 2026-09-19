package dashboard

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
)

func TestDashboardE2EStagesWritableRuntimeConfig(t *testing.T) {
	deployment := loadDashboardDeployment(t)

	initContainer := requireContainer(t, deployment.Spec.Template.Spec.InitContainers, "stage-dashboard-runtime-config")
	requireVolumeMount(t, initContainer.VolumeMounts, "router-config-seed", true)
	requireVolumeMount(t, initContainer.VolumeMounts, "router-config-runtime", false)

	dashboard := requireContainer(t, deployment.Spec.Template.Spec.Containers, "dashboard")
	requireVolumeMount(t, dashboard.VolumeMounts, "router-config-runtime", false)
	if dashboard.ReadinessProbe == nil || dashboard.ReadinessProbe.HTTPGet == nil || dashboard.ReadinessProbe.HTTPGet.Path != "/healthz" {
		t.Fatal("Dashboard replacement must not be ready before its HTTP handlers initialize")
	}
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

func TestDashboardE2ESeparatesBenchmarkOwnerAndPersistentStore(t *testing.T) {
	deployment := loadDashboardDeployment(t)
	dashboard := requireContainer(t, deployment.Spec.Template.Spec.Containers, "dashboard")
	requireVolumeMount(t, dashboard.VolumeMounts, "dashboard-data", false)
	if value := envValue(dashboard.Env, "SR_BENCH_URL"); value != "http://semantic-router-sr-bench:8090" {
		t.Fatalf("Dashboard must proxy the independent worker, got %q", value)
	}
	if envValue(dashboard.Env, "EVALUATION_DATA_DIR") != "" {
		t.Fatal("retired Evaluation Plane must not own the benchmark store")
	}
	var workerDeployment appsv1.Deployment
	loadSrBenchResource(t, "Deployment", &workerDeployment)
	worker := requireContainer(t, workerDeployment.Spec.Template.Spec.Containers, "worker")
	if worker.Image != dashboard.Image {
		t.Fatal("worker must exercise the product service packaged in the same built image")
	}
	if workerDeployment.Name == deployment.Name || reflect.DeepEqual(workerDeployment.Spec.Selector.MatchLabels, deployment.Spec.Selector.MatchLabels) {
		t.Fatal("worker lifecycle must be independent of the Dashboard pod")
	}
	if envValue(worker.Env, "SR_BENCH_TOKEN") == "" || envValue(worker.Env, "SR_BENCH_TOKEN") != envValue(dashboard.Env, "SR_BENCH_TOKEN") {
		t.Fatal("Dashboard and worker must share service authentication")
	}
	if !reflect.DeepEqual(worker.Command, []string{"python3", "-m", "cli.main"}) || !reflect.DeepEqual(worker.Args, []string{"benchmark", "--store", "/data/store", "serve", "--host", "0.0.0.0", "--port", "8090"}) {
		t.Fatal("worker must start the actual public CLI service with the durable store")
	}
	requireVolumeMount(t, worker.VolumeMounts, "benchmark-data", false)
	var workerPVC corev1.PersistentVolumeClaim
	loadSrBenchResource(t, "PersistentVolumeClaim", &workerPVC)
	if workerPVC.Labels["vllm.ai/e2e-managed"] != "true" {
		t.Fatal("benchmark PVC must be scoped to profile cleanup")
	}
	if len(workerDeployment.Spec.Template.Spec.Volumes) != 2 || workerDeployment.Spec.Template.Spec.Volumes[0].PersistentVolumeClaim.ClaimName != workerPVC.Name {
		t.Fatal("worker must mount its independent PVC")
	}
	for _, volume := range deployment.Spec.Template.Spec.Volumes {
		if volume.PersistentVolumeClaim != nil && volume.PersistentVolumeClaim.ClaimName == workerPVC.Name {
			t.Fatal("Dashboard must not mount the benchmark store")
		}
	}
	var service corev1.Service
	loadSrBenchResource(t, "Service", &service)
	if !reflect.DeepEqual(service.Spec.Selector, workerDeployment.Spec.Selector.MatchLabels) {
		t.Fatal("benchmark service must select its worker deployment")
	}
	if worker.ReadinessProbe == nil || worker.ReadinessProbe.Exec == nil {
		t.Fatal("worker readiness must check authenticated service health")
	}
	assertDashboardPersistentVolume(t, deployment)
}

func assertDashboardPersistentVolume(t *testing.T, deployment appsv1.Deployment) {
	t.Helper()
	for _, volume := range deployment.Spec.Template.Spec.Volumes {
		if volume.Name == "dashboard-data" && volume.PersistentVolumeClaim != nil {
			return
		}
	}
	t.Fatal("dashboard-data must be a persistent volume claim")
}

func loadSrBenchResource(t *testing.T, kind string, result any) {
	t.Helper()
	raw, err := os.ReadFile("sr-bench-deployment.yaml")
	if err != nil {
		t.Fatal(err)
	}
	decoder := utilyaml.NewYAMLToJSONDecoder(bytes.NewReader(raw))
	for {
		var document json.RawMessage
		if err := decoder.Decode(&document); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		var meta struct{ Kind string }
		if err := json.Unmarshal(document, &meta); err != nil {
			t.Fatal(err)
		}
		if meta.Kind == kind {
			if err := json.Unmarshal(document, result); err != nil {
				t.Fatal(err)
			}
			return
		}
	}
	t.Fatalf("missing %s in benchmark fixture", kind)
}

func loadDashboardDeployment(t *testing.T) appsv1.Deployment {
	t.Helper()
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
	return deployment
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
