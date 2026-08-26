package dashboard

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
)

type dashboardEgressPolicy struct {
	Schemes []string              `json:"schemes"`
	Hosts   []dashboardEgressHost `json:"hosts"`
}

type dashboardEgressHost struct {
	Host       string   `json:"host"`
	Ports      []int    `json:"ports"`
	AllowCIDRs []string `json:"allow_cidrs"`
}

func TestDashboardE2EOnlyMountsRouterBootstrapConfigReadOnly(t *testing.T) {
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

	if len(deployment.Spec.Template.Spec.InitContainers) != 0 {
		t.Fatalf("dashboard must not stage a writable Router config: %#v", deployment.Spec.Template.Spec.InitContainers)
	}

	dashboard := requireContainer(t, deployment.Spec.Template.Spec.Containers, "dashboard")
	requireVolumeMount(t, dashboard.VolumeMounts, "router-config", true)
	if value := envValue(dashboard.Env, "DASHBOARD_RUNTIME_CONFIG_WRITABLE"); value != "" {
		t.Fatalf("removed DASHBOARD_RUNTIME_CONFIG_WRITABLE = %q", value)
	}

	for _, volume := range deployment.Spec.Template.Spec.Volumes {
		if volume.Name == "router-config" && volume.ConfigMap != nil {
			if volume.EmptyDir != nil {
				t.Fatal("router-config must not be writable emptyDir state")
			}
			return
		}
	}
	t.Fatal("router-config ConfigMap volume not found")
}

func TestDashboardSetupPublishesBeforeWaitingForRouterReadiness(t *testing.T) {
	profile := NewProfile()
	steps := profile.setupSteps()
	want := []string{
		setupStepManagedPrerequisites,
		setupStepGatewayFoundations,
		setupStepPublicInference,
		setupStepDashboardClient,
		setupStepFirstPublication,
		setupStepRouterReadiness,
	}
	if len(steps) != len(want) {
		t.Fatalf("setup steps = %d, want %d", len(steps), len(want))
	}
	for index, expected := range want {
		if steps[index].name != expected {
			t.Fatalf("setup step %d = %q, want %q", index, steps[index].name, expected)
		}
	}
}

func TestDashboardKubectlArgsKeepGlobalFlagsBeforeExecCommand(t *testing.T) {
	arguments := dashboardKubectlArgs(
		"/tmp/e2e-kubeconfig",
		"exec",
		"deployment/semantic-router-dashboard",
		"--",
		"test",
		"!",
		"-e",
		"/run/bootstrap-token",
	)
	want := []string{
		"--kubeconfig",
		"/tmp/e2e-kubeconfig",
		"exec",
		"deployment/semantic-router-dashboard",
		"--",
		"test",
		"!",
		"-e",
		"/run/bootstrap-token",
	}
	if !reflect.DeepEqual(arguments, want) {
		t.Fatalf("kubectl arguments = %#v, want %#v", arguments, want)
	}
}

func TestDashboardRouterReplicaContractMatchesProfileValues(t *testing.T) {
	raw, err := os.ReadFile(filepath.Base(valuesFile))
	if err != nil {
		t.Fatalf("read Dashboard values: %v", err)
	}
	jsonDocument, err := utilyaml.ToJSON(raw)
	if err != nil {
		t.Fatalf("convert Dashboard values to JSON: %v", err)
	}
	var values struct {
		ReplicaCount   int32 `json:"replicaCount"`
		ConfigOverride struct {
			Global struct {
				Services struct {
					Agent struct {
						PublicInferenceEndpoint string `json:"public_inference_endpoint"`
					} `json:"agent"`
				} `json:"services"`
			} `json:"global"`
		} `json:"configOverride"`
	}
	if err := json.Unmarshal(jsonDocument, &values); err != nil {
		t.Fatalf("decode Dashboard values: %v", err)
	}
	if values.ReplicaCount != semanticRouterReplicaCount {
		t.Fatalf(
			"Dashboard Router replicaCount = %d, readiness contract = %d",
			values.ReplicaCount,
			semanticRouterReplicaCount,
		)
	}
	wantEndpoint := "http://" + dashboardPublicInferenceServiceName + "." + namespaceRouter + ".svc.cluster.local/v1/chat/completions"
	if got := values.ConfigOverride.Global.Services.Agent.PublicInferenceEndpoint; got != wantEndpoint {
		t.Fatalf("Agent public inference endpoint = %q, want %q", got, wantEndpoint)
	}
}

func TestDashboardBootstrapTokenContractMatchesDeployment(t *testing.T) {
	raw, err := os.ReadFile(filepath.Base(dashboardE2EDeploymentManifest))
	if err != nil {
		t.Fatalf("read Dashboard deployment: %v", err)
	}
	jsonDocument, err := utilyaml.ToJSON(raw)
	if err != nil {
		t.Fatalf("convert Dashboard deployment to JSON: %v", err)
	}
	var deployment appsv1.Deployment
	if err := json.Unmarshal(jsonDocument, &deployment); err != nil {
		t.Fatalf("decode Dashboard deployment: %v", err)
	}
	dashboard := requireContainer(t, deployment.Spec.Template.Spec.Containers, "dashboard")
	if value := envValue(dashboard.Env, "DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE"); value != dashboardBootstrapTokenPath {
		t.Fatalf("Dashboard bootstrap token path = %q, consumption check = %q", value, dashboardBootstrapTokenPath)
	}
}

func TestDashboardIdentityFilesUsePinnedPrivateSubPathMounts(t *testing.T) {
	deployment := readDashboardDeploymentForTest(t)
	wantMounts := map[string]string{
		"/run/vllm-sr-dashboard-assertion-signing-key.pem": "assertion-signing-key.pem",
		"/run/vllm-sr-dashboard-issuer-tls.crt":            "tls.crt",
		"/run/vllm-sr-dashboard-issuer-tls.key":            "tls.key",
		"/run/vllm-sr-dashboard-ca.crt":                    "ca.crt",
	}
	dashboard := requireContainer(t, deployment.Spec.Template.Spec.Containers, "dashboard")
	wantEnvironment := map[string]string{
		"DASHBOARD_SIGNING_KEY_FILE":     "/run/vllm-sr-dashboard-assertion-signing-key.pem",
		"DASHBOARD_ISSUER_TLS_CERT_FILE": "/run/vllm-sr-dashboard-issuer-tls.crt",
		"DASHBOARD_ISSUER_TLS_KEY_FILE":  "/run/vllm-sr-dashboard-issuer-tls.key",
		"SSL_CERT_FILE":                  "/run/vllm-sr-dashboard-ca.crt",
	}
	assertDashboardIdentityEnvironment(t, dashboard, wantEnvironment)
	assertDashboardIdentityMounts(t, dashboard.VolumeMounts, wantMounts)
	assertDashboardIdentityVolume(t, deployment.Spec.Template.Spec.Volumes)
}

func readDashboardDeploymentForTest(t *testing.T) appsv1.Deployment {
	t.Helper()
	raw, err := os.ReadFile(filepath.Base(dashboardE2EDeploymentManifest))
	if err != nil {
		t.Fatalf("read Dashboard deployment: %v", err)
	}
	jsonDocument, err := utilyaml.ToJSON(raw)
	if err != nil {
		t.Fatalf("convert Dashboard deployment to JSON: %v", err)
	}
	var deployment appsv1.Deployment
	if err := json.Unmarshal(jsonDocument, &deployment); err != nil {
		t.Fatalf("decode Dashboard deployment: %v", err)
	}
	return deployment
}

func assertDashboardIdentityEnvironment(
	t *testing.T,
	dashboard corev1.Container,
	wantEnvironment map[string]string,
) {
	t.Helper()
	for name, wantPath := range wantEnvironment {
		if got := envValue(dashboard.Env, name); got != wantPath {
			t.Fatalf("Dashboard %s = %q, want %q", name, got, wantPath)
		}
	}
}

func assertDashboardIdentityMounts(
	t *testing.T,
	mounts []corev1.VolumeMount,
	wantMounts map[string]string,
) {
	t.Helper()
	for _, mount := range mounts {
		if mount.Name != "dashboard-identity" {
			continue
		}
		wantSubPath, ok := wantMounts[mount.MountPath]
		if !ok {
			t.Fatalf("unexpected Dashboard identity mount path %q", mount.MountPath)
		}
		if mount.SubPath != wantSubPath {
			t.Fatalf("Dashboard identity mount %q subPath = %q, want %q", mount.MountPath, mount.SubPath, wantSubPath)
		}
		if !mount.ReadOnly {
			t.Fatalf("Dashboard identity mount %q is writable", mount.MountPath)
		}
		if parent := filepath.Dir(mount.MountPath); parent != "/run" {
			t.Fatalf("Dashboard identity mount parent = %q, want /run", parent)
		}
		delete(wantMounts, mount.MountPath)
	}
	if len(wantMounts) != 0 {
		t.Fatalf("missing Dashboard identity mounts: %#v", wantMounts)
	}
}

func assertDashboardIdentityVolume(t *testing.T, volumes []corev1.Volume) {
	t.Helper()
	for _, volume := range volumes {
		if volume.Name != "dashboard-identity" {
			continue
		}
		if volume.Secret == nil || volume.Secret.DefaultMode == nil {
			t.Fatal("Dashboard identity volume must be a private Secret volume")
		}
		if got := *volume.Secret.DefaultMode; got != 0o400 {
			t.Fatalf("Dashboard identity Secret mode = %#o, want 0400", got)
		}
		return
	}
	t.Fatal("Dashboard identity Secret volume not found")
}

func TestDashboardIdentitySecretIsImmutable(t *testing.T) {
	secret := newDashboardIdentitySecret(map[string][]byte{"tls.crt": []byte("test")})
	if secret.Immutable == nil || !*secret.Immutable {
		t.Fatal("Dashboard identity Secret must be immutable for pinned subPath mounts")
	}
	if secret.Namespace != namespaceRouter || secret.Name != dashboardIdentitySecretName {
		t.Fatalf("Dashboard identity Secret = %s/%s", secret.Namespace, secret.Name)
	}
}

func TestDashboardIdentitySecretRequiresCleanProfileReplacement(t *testing.T) {
	secret := newDashboardIdentitySecret(map[string][]byte{"tls.crt": []byte("test")})
	err := validateE2ESecretReplacement(secret)
	want := "managed Secret vllm-semantic-router-system/semantic-router-dashboard-e2e-dashboard is immutable; rerun the profile in a clean E2E cluster"
	if err == nil || err.Error() != want {
		t.Fatalf("validateE2ESecretReplacement() error = %v, want %q", err, want)
	}
}

func TestDashboardRouterEgressAllowsExactManagedIssuer(t *testing.T) {
	policy := readDashboardEgressPolicy(t)
	if len(policy.Schemes) != 2 || policy.Schemes[0] != "http" || policy.Schemes[1] != "https" {
		t.Fatalf("Dashboard Router egress schemes = %#v, want [http https]", policy.Schemes)
	}
	issuer, found := dashboardIssuerEgress(policy.Hosts)
	if !found {
		t.Fatalf("Dashboard Router egress policy does not allow exact issuer %q", dashboardIssuerDNS)
	}
	if len(issuer.Ports) != 1 || issuer.Ports[0] != 9443 {
		t.Fatalf("Dashboard issuer egress ports = %#v, want [9443]", issuer.Ports)
	}
	if len(issuer.AllowCIDRs) == 0 {
		t.Fatal("Dashboard issuer egress must admit the disposable cluster network")
	}
}

func readDashboardEgressPolicy(t *testing.T) dashboardEgressPolicy {
	t.Helper()
	raw, err := os.ReadFile(filepath.Base(dashboardE2EEgressManifest))
	if err != nil {
		t.Fatalf("read Dashboard Router egress policy: %v", err)
	}
	jsonDocument, err := utilyaml.ToJSON(raw)
	if err != nil {
		t.Fatalf("convert Dashboard Router egress ConfigMap to JSON: %v", err)
	}
	var configMap corev1.ConfigMap
	if err := json.Unmarshal(jsonDocument, &configMap); err != nil {
		t.Fatalf("decode Dashboard Router egress ConfigMap: %v", err)
	}
	policyDocument, err := utilyaml.ToJSON([]byte(configMap.Data["policy.yaml"]))
	if err != nil {
		t.Fatalf("convert Dashboard Router egress policy to JSON: %v", err)
	}
	var policy dashboardEgressPolicy
	if err := json.Unmarshal(policyDocument, &policy); err != nil {
		t.Fatalf("decode Dashboard Router egress policy: %v", err)
	}
	return policy
}

func dashboardIssuerEgress(hosts []dashboardEgressHost) (dashboardEgressHost, bool) {
	for _, host := range hosts {
		if host.Host != dashboardIssuerDNS {
			continue
		}
		return host, true
	}
	return dashboardEgressHost{}, false
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
