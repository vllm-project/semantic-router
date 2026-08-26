package dashboard

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
)

func TestRouterBootstrapConfigMapNameFollowsRouterDeployment(t *testing.T) {
	deployment := &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{Volumes: []corev1.Volume{
			{Name: "other"},
			{Name: routerBootstrapConfigVolume, VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: "semantic-router-config-a1b2c3"},
			}}},
		}},
	}}}

	name, err := routerBootstrapConfigMapName(deployment)
	if err != nil {
		t.Fatalf("routerBootstrapConfigMapName() error = %v", err)
	}
	if name != "semantic-router-config-a1b2c3" {
		t.Fatalf("routerBootstrapConfigMapName() = %q", name)
	}
}

func TestRouterBootstrapConfigMapNameRejectsMissingVolume(t *testing.T) {
	if _, err := routerBootstrapConfigMapName(&appsv1.Deployment{}); err == nil {
		t.Fatal("routerBootstrapConfigMapName() accepted a deployment without the Router config volume")
	}
}

func TestRouterBootstrapConfigMapNameRejectsMalformedVolume(t *testing.T) {
	deployment := &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{Volumes: []corev1.Volume{{Name: routerBootstrapConfigVolume}}},
	}}}
	if _, err := routerBootstrapConfigMapName(deployment); err == nil {
		t.Fatal("routerBootstrapConfigMapName() accepted a non-ConfigMap Router config volume")
	}
}

func TestRouterBootstrapConfigMapNameRejectsAmbiguousVolume(t *testing.T) {
	source := func(name string) corev1.Volume {
		return corev1.Volume{
			Name: routerBootstrapConfigVolume,
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: name},
			}},
		}
	}
	deployment := &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{Volumes: []corev1.Volume{source("first"), source("second")}},
	}}}
	if _, err := routerBootstrapConfigMapName(deployment); err == nil {
		t.Fatal("routerBootstrapConfigMapName() accepted duplicate Router config volumes")
	}
}

func TestLoadDashboardDeploymentBindsInstalledRouterConfig(t *testing.T) {
	deployment, err := loadDashboardDeployment(
		filepath.Base(dashboardE2EDeploymentManifest),
		"semantic-router-config-a1b2c3",
	)
	if err != nil {
		t.Fatalf("loadDashboardDeployment() error = %v", err)
	}
	if deployment.Namespace != namespaceRouter {
		t.Fatalf("Dashboard namespace = %q, want %q", deployment.Namespace, namespaceRouter)
	}
	for _, volume := range deployment.Spec.Template.Spec.Volumes {
		if volume.Name == dashboardRouterConfigVolume && volume.ConfigMap != nil {
			if volume.ConfigMap.Name != "semantic-router-config-a1b2c3" {
				t.Fatalf("Dashboard Router ConfigMap = %q", volume.ConfigMap.Name)
			}
			return
		}
	}
	t.Fatalf("Dashboard deployment has no %q volume", dashboardRouterConfigVolume)
}

func TestBindDashboardRouterConfigRejectsStaleStaticName(t *testing.T) {
	deployment := &appsv1.Deployment{Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{Volumes: []corev1.Volume{{
			Name: dashboardRouterConfigVolume,
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: "semantic-router-config"},
			}},
		}}},
	}}}

	if err := bindDashboardRouterConfig(deployment, "semantic-router-config-a1b2c3"); err == nil {
		t.Fatal("bindDashboardRouterConfig() accepted a stale static ConfigMap name")
	}
}

func TestApplyDashboardDeploymentRejectsMissingRouterConfigMap(t *testing.T) {
	fixture := &dashboardKubeAPIFixture{
		router: dashboardTestRouterDeployment("semantic-router-config-a1b2c3"),
	}
	err := applyDashboardDeploymentWithFixture(t, fixture)
	if err == nil || !strings.Contains(err.Error(), "read Router bootstrap ConfigMap") {
		t.Fatalf("applyDashboardDeployment() error = %v, want missing ConfigMap", err)
	}
}

func TestApplyDashboardDeploymentRejectsMutableRouterConfigMap(t *testing.T) {
	fixture := &dashboardKubeAPIFixture{
		router:    dashboardTestRouterDeployment("semantic-router-config-a1b2c3"),
		configMap: dashboardTestRouterConfigMap("semantic-router-config-a1b2c3", false),
	}
	err := applyDashboardDeploymentWithFixture(t, fixture)
	if err == nil || !strings.Contains(err.Error(), "must be immutable") {
		t.Fatalf("applyDashboardDeployment() error = %v, want immutable ConfigMap rejection", err)
	}
}

func TestApplyDashboardDeploymentCreatesManagedBoundDeployment(t *testing.T) {
	fixture := &dashboardKubeAPIFixture{
		router:    dashboardTestRouterDeployment("semantic-router-config-a1b2c3"),
		configMap: dashboardTestRouterConfigMap("semantic-router-config-a1b2c3", true),
	}
	if err := applyDashboardDeploymentWithFixture(t, fixture); err != nil {
		t.Fatalf("applyDashboardDeployment() error = %v", err)
	}
	if fixture.created == nil {
		t.Fatal("applyDashboardDeployment() did not create the Dashboard deployment")
	}
	if fixture.created.Labels["vllm.ai/e2e-managed"] != "true" {
		t.Fatalf("created Dashboard labels = %#v", fixture.created.Labels)
	}
	for _, volume := range fixture.created.Spec.Template.Spec.Volumes {
		if volume.Name == dashboardRouterConfigVolume && volume.ConfigMap != nil {
			if volume.ConfigMap.Name != "semantic-router-config-a1b2c3" {
				t.Fatalf("created Dashboard Router ConfigMap = %q", volume.ConfigMap.Name)
			}
			return
		}
	}
	t.Fatalf("created Dashboard deployment has no %q volume", dashboardRouterConfigVolume)
}

func TestApplyDashboardDeploymentRefusesUnmanagedExistingDeployment(t *testing.T) {
	fixture := &dashboardKubeAPIFixture{
		router:    dashboardTestRouterDeployment("semantic-router-config-a1b2c3"),
		configMap: dashboardTestRouterConfigMap("semantic-router-config-a1b2c3", true),
		existing: &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{
			Name: deploymentDashboard, Namespace: namespaceRouter,
		}},
	}
	err := applyDashboardDeploymentWithFixture(t, fixture)
	if err == nil || !strings.Contains(err.Error(), "refuse to replace unmanaged") {
		t.Fatalf("applyDashboardDeployment() error = %v, want unmanaged Deployment rejection", err)
	}
	if fixture.created != nil {
		t.Fatal("applyDashboardDeployment() replaced an unmanaged Deployment")
	}
}

type dashboardKubeAPIFixture struct {
	router    *appsv1.Deployment
	configMap *corev1.ConfigMap
	existing  *appsv1.Deployment
	created   *appsv1.Deployment
}

func applyDashboardDeploymentWithFixture(t *testing.T, fixture *dashboardKubeAPIFixture) error {
	t.Helper()
	repositoryRoot, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatalf("resolve repository root: %v", err)
	}
	t.Chdir(repositoryRoot)
	server := httptest.NewServer(fixture)
	t.Cleanup(server.Close)
	client, err := kubernetes.NewForConfig(&rest.Config{Host: server.URL})
	if err != nil {
		t.Fatalf("create Kubernetes client: %v", err)
	}
	return (&Profile{}).applyDashboardDeployment(context.Background(), &framework.SetupOptions{KubeClient: client})
}

func dashboardTestRouterDeployment(configMapName string) *appsv1.Deployment {
	return &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name: "semantic-router", Namespace: namespaceRouter,
		},
		Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Volumes: []corev1.Volume{{
				Name: routerBootstrapConfigVolume,
				VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
					LocalObjectReference: corev1.LocalObjectReference{Name: configMapName},
				}},
			}},
		}}},
	}
}

func dashboardTestRouterConfigMap(name string, immutable bool) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "ConfigMap"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: namespaceRouter,
		},
		Immutable: boolPointer(immutable),
	}
}

func (fixture *dashboardKubeAPIFixture) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	switch request.Method {
	case http.MethodGet:
		fixture.serveGet(writer, request)
	case http.MethodPost:
		fixture.servePost(writer, request)
	default:
		fixture.writeUnexpectedRequest(writer, request)
	}
}

func (fixture *dashboardKubeAPIFixture) serveGet(writer http.ResponseWriter, request *http.Request) {
	const (
		deploymentsPath = "/apis/apps/v1/namespaces/" + namespaceRouter + "/deployments"
		configMapsPath  = "/api/v1/namespaces/" + namespaceRouter + "/configmaps/"
	)
	switch {
	case request.URL.Path == deploymentsPath+"/semantic-router":
		fixture.writeObject(writer, http.StatusOK, fixture.router)
	case strings.HasPrefix(request.URL.Path, configMapsPath):
		name := strings.TrimPrefix(request.URL.Path, configMapsPath)
		fixture.serveConfigMap(writer, name)
	case request.URL.Path == deploymentsPath+"/"+deploymentDashboard:
		fixture.serveExistingDeployment(writer)
	default:
		fixture.writeUnexpectedRequest(writer, request)
	}
}

func (fixture *dashboardKubeAPIFixture) serveConfigMap(writer http.ResponseWriter, name string) {
	if fixture.configMap == nil || fixture.configMap.Name != name {
		fixture.writeNotFound(writer, "configmaps", name)
		return
	}
	fixture.writeObject(writer, http.StatusOK, fixture.configMap)
}

func (fixture *dashboardKubeAPIFixture) serveExistingDeployment(writer http.ResponseWriter) {
	if fixture.existing == nil {
		fixture.writeNotFound(writer, "deployments", deploymentDashboard)
		return
	}
	fixture.writeObject(writer, http.StatusOK, fixture.existing)
}

func (fixture *dashboardKubeAPIFixture) servePost(writer http.ResponseWriter, request *http.Request) {
	const deploymentsPath = "/apis/apps/v1/namespaces/" + namespaceRouter + "/deployments"
	if request.URL.Path != deploymentsPath {
		fixture.writeUnexpectedRequest(writer, request)
		return
	}
	var deployment appsv1.Deployment
	if err := json.NewDecoder(request.Body).Decode(&deployment); err != nil {
		fixture.writeFailure(writer, http.StatusBadRequest, err.Error())
		return
	}
	fixture.created = deployment.DeepCopy()
	fixture.writeObject(writer, http.StatusCreated, &deployment)
}

func (fixture *dashboardKubeAPIFixture) writeUnexpectedRequest(
	writer http.ResponseWriter,
	request *http.Request,
) {
	fixture.writeFailure(
		writer,
		http.StatusInternalServerError,
		fmt.Sprintf("unexpected Kubernetes request %s %s", request.Method, request.URL.Path),
	)
}

func (fixture *dashboardKubeAPIFixture) writeNotFound(writer http.ResponseWriter, resource string, name string) {
	status := &metav1.Status{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Status"},
		Status:   metav1.StatusFailure,
		Reason:   metav1.StatusReasonNotFound,
		Message:  fmt.Sprintf("%s %q not found", resource, name),
		Code:     http.StatusNotFound,
	}
	fixture.writeObject(writer, http.StatusNotFound, status)
}

func (fixture *dashboardKubeAPIFixture) writeFailure(writer http.ResponseWriter, code int, message string) {
	status := &metav1.Status{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Status"},
		Status:   metav1.StatusFailure,
		Reason:   metav1.StatusReasonInternalError,
		Message:  message,
		Code:     int32(code),
	}
	fixture.writeObject(writer, code, status)
}

func (fixture *dashboardKubeAPIFixture) writeObject(writer http.ResponseWriter, code int, value interface{}) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(code)
	_ = json.NewEncoder(writer).Encode(value)
}
