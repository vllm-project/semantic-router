/*
Copyright 2026 vLLM Semantic Router Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"context"
	"reflect"
	"testing"

	"github.com/go-logr/logr"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

var testDurableBootstrap = bootstrapDeploymentContract{
	Revision:             "sha256:durable-test",
	ManagementStore:      true,
	RuntimeStore:         true,
	ManagementAPIEnabled: true,
	AccessEnabled:        true,
	ManagementPort:       9443,
	BackendDispatchPort:  8181,
	PostgresDSNEnv:       "TEST_DATABASE_URL",
}

func TestManagementMigrationJobUsesExplicitDeploymentInputs(t *testing.T) {
	createServiceAccount := true
	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Image:          vllmv1alpha1.ImageSpec{Repository: "example.invalid/router", Tag: "v1.0.0"},
			ServiceAccount: vllmv1alpha1.ServiceAccountSpec{Create: &createServiceAccount},
			Env: []corev1.EnvVar{{Name: "TEST_DATABASE_URL", ValueFrom: &corev1.EnvVarSource{
				SecretKeyRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "management-store"}, Key: "dsn"},
			}}},
			EnvFrom: []corev1.EnvFromSource{{SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: "router-runtime"},
			}}},
			Volumes:      []corev1.Volume{{Name: "management-secrets"}},
			VolumeMounts: []corev1.VolumeMount{{Name: "management-secrets", MountPath: "/run/secrets/router", ReadOnly: true}},
		},
	}

	job, err := (&SemanticRouterReconciler{}).generateMigrationJob(sr, testDurableBootstrap)
	if err != nil {
		t.Fatalf("generateMigrationJob() error = %v", err)
	}
	if job.Spec.Template.Spec.ServiceAccountName != "router" {
		t.Fatalf("migration ServiceAccount = %q", job.Spec.Template.Spec.ServiceAccountName)
	}
	container := job.Spec.Template.Spec.Containers[0]
	if container.Image != "example.invalid/router:v1.0.0" {
		t.Fatalf("migration image = %q", container.Image)
	}
	if want := []string{"--dsn-env", "TEST_DATABASE_URL", "--timeout", "5m"}; !reflect.DeepEqual(container.Args, want) {
		t.Fatalf("migration args = %#v, want %#v", container.Args, want)
	}
	if !reflect.DeepEqual(container.Env, sr.Spec.Env[:1]) || len(container.EnvFrom) != 0 ||
		len(container.VolumeMounts) != 0 || len(job.Spec.Template.Spec.Volumes) != 0 {
		t.Fatal("migration Job received inputs beyond its PostgreSQL DSN Secret key")
	}

	revised := sr.DeepCopy()
	revised.Spec.Env[0].ValueFrom.SecretKeyRef.Name = "management-store-v2"
	revisedJob, err := (&SemanticRouterReconciler{}).generateMigrationJob(revised, testDurableBootstrap)
	if err != nil {
		t.Fatalf("generate revised migration Job: %v", err)
	}
	if revisedJob.Name == job.Name {
		t.Fatalf("migration Job name %q did not change with its deployment inputs", job.Name)
	}

	doNotCreateServiceAccount := false
	sr.Spec.ServiceAccount.Create = &doNotCreateServiceAccount
	if got := serviceAccountName(sr); got != "" {
		t.Fatalf("disabled ServiceAccount name = %q, want the namespace default", got)
	}
}

func TestManagementMigrationProjectsOneEnvFromSecretKey(t *testing.T) {
	sr := &vllmv1alpha1.SemanticRouter{ObjectMeta: metav1.ObjectMeta{Name: "router"}, Spec: vllmv1alpha1.SemanticRouterSpec{
		EnvFrom: []corev1.EnvFromSource{{SecretRef: &corev1.SecretEnvSource{
			LocalObjectReference: corev1.LocalObjectReference{Name: "router-runtime"},
		}}},
	}}

	job, err := (&SemanticRouterReconciler{}).generateMigrationJob(sr, testDurableBootstrap)
	if err != nil {
		t.Fatalf("generateMigrationJob() error = %v", err)
	}
	container := job.Spec.Template.Spec.Containers[0]
	if len(container.Env) != 1 || container.Env[0].Name != testDurableBootstrap.PostgresDSNEnv {
		t.Fatalf("migration environment = %#v", container.Env)
	}
	secret := container.Env[0].ValueFrom.SecretKeyRef
	if secret == nil || secret.Name != "router-runtime" || secret.Key != testDurableBootstrap.PostgresDSNEnv {
		t.Fatalf("migration DSN projection = %#v", secret)
	}
	if len(container.EnvFrom) != 0 {
		t.Fatalf("migration inherited envFrom = %#v", container.EnvFrom)
	}
}

func TestManagementMigrationRejectsAmbiguousEnvFromSecrets(t *testing.T) {
	sr := &vllmv1alpha1.SemanticRouter{ObjectMeta: metav1.ObjectMeta{Name: "router"}, Spec: vllmv1alpha1.SemanticRouterSpec{
		EnvFrom: []corev1.EnvFromSource{
			{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "first"}}},
			{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "second"}}},
		},
	}}

	if _, err := (&SemanticRouterReconciler{}).generateMigrationJob(sr, testDurableBootstrap); err == nil {
		t.Fatal("ambiguous migration envFrom Secrets were accepted")
	}
}

func TestManagementMigrationProjectsOnlyDSNFileSecret(t *testing.T) {
	bootstrap := testDurableBootstrap
	bootstrap.PostgresDSNEnv = ""
	bootstrap.PostgresDSNFile = "/run/secrets/management/postgres-dsn"
	sr := &vllmv1alpha1.SemanticRouter{ObjectMeta: metav1.ObjectMeta{Name: "router"}, Spec: vllmv1alpha1.SemanticRouterSpec{
		Volumes: []corev1.Volume{
			{Name: "management", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "management-store"}}},
			{Name: "runtime", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "router-runtime"}}},
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: "management", MountPath: "/run/secrets/management", ReadOnly: true},
			{Name: "runtime", MountPath: "/run/secrets/runtime", ReadOnly: true},
		},
	}}

	job, err := (&SemanticRouterReconciler{}).generateMigrationJob(sr, bootstrap)
	if err != nil {
		t.Fatalf("generateMigrationJob() error = %v", err)
	}
	container := job.Spec.Template.Spec.Containers[0]
	if len(container.Env) != 0 || len(container.VolumeMounts) != 1 || container.VolumeMounts[0].Name != "management" {
		t.Fatalf("migration container inputs = env %#v mounts %#v", container.Env, container.VolumeMounts)
	}
	if len(job.Spec.Template.Spec.Volumes) != 1 || job.Spec.Template.Spec.Volumes[0].Name != "management" {
		t.Fatalf("migration volumes = %#v", job.Spec.Template.Spec.Volumes)
	}
}

func TestDurableServicesSeparateInferenceAndManagement(t *testing.T) {
	r := &SemanticRouterReconciler{}
	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"},
		Spec: vllmv1alpha1.SemanticRouterSpec{Service: vllmv1alpha1.ServiceSpec{
			Management: vllmv1alpha1.ManagementServiceSpec{Port: 8443},
		}},
	}
	services := r.generateServices(sr, gatewayModeExternal, testDurableBootstrap)
	if len(services) != 4 {
		t.Fatalf("durable Service count = %d, want 4", len(services))
	}
	byName := indexServicesByName(services)
	requirePublicInferenceService(t, byName["router"])
	requireManagementService(t, byName["router-management"])
	requireBackendDispatchService(t, byName["router-backend-dispatch"])
	requireMetricsService(t, byName["router-metrics"])
}

func indexServicesByName(services []*corev1.Service) map[string]*corev1.Service {
	byName := make(map[string]*corev1.Service, len(services))
	for _, service := range services {
		byName[service.Name] = service
	}
	return byName
}

func requirePublicInferenceService(t *testing.T, service *corev1.Service) {
	t.Helper()
	if service == nil {
		t.Fatal("public inference Service is missing")
	}
	if len(service.Spec.Ports) != 1 {
		t.Fatalf("public inference Service ports = %#v", service.Spec.Ports)
	}
	if service.Spec.Ports[0].Name != "grpc" {
		t.Fatalf("public inference Service port = %#v", service.Spec.Ports[0])
	}
	if service.Spec.PublishNotReadyAddresses {
		t.Fatal("public inference Service must remain readiness-gated")
	}
}

func requireManagementService(t *testing.T, service *corev1.Service) {
	t.Helper()
	if service == nil {
		t.Fatal("private Management Service is missing")
	}
	if service.Spec.Type != corev1.ServiceTypeClusterIP {
		t.Fatalf("private Management Service type = %q", service.Spec.Type)
	}
	if len(service.Spec.Ports) != 1 {
		t.Fatalf("private Management Service ports = %#v", service.Spec.Ports)
	}
	port := service.Spec.Ports[0]
	if port.Port != 8443 {
		t.Fatalf("private Management Service port = %d, want 8443", port.Port)
	}
	if port.TargetPort.IntVal != 9443 {
		t.Fatalf("private Management target port = %d, want 9443", port.TargetPort.IntVal)
	}
	if !service.Spec.PublishNotReadyAddresses {
		t.Fatal("private Management Service must be reachable before inference readiness")
	}
}

func requireBackendDispatchService(t *testing.T, service *corev1.Service) {
	t.Helper()
	if service == nil {
		t.Fatal("backend dispatch Service is missing")
	}
	if len(service.Spec.Ports) != 1 {
		t.Fatalf("backend dispatch Service ports = %#v", service.Spec.Ports)
	}
	if service.Spec.Ports[0].Name != backendDispatchPortName {
		t.Fatalf("backend dispatch Service port = %#v", service.Spec.Ports[0])
	}
	if service.Spec.PublishNotReadyAddresses {
		t.Fatal("backend dispatch Service must remain readiness-gated")
	}
}

func requireMetricsService(t *testing.T, service *corev1.Service) {
	t.Helper()
	if service == nil {
		t.Fatal("metrics Service is missing")
	}
	if service.Spec.PublishNotReadyAddresses {
		t.Fatal("metrics Service must remain readiness-gated")
	}
}

func TestDurableDeploymentUsesHTTPSReadinessAndTopologySpread(t *testing.T) {
	r := &SemanticRouterReconciler{}
	sr := &vllmv1alpha1.SemanticRouter{ObjectMeta: metav1.ObjectMeta{Name: "router"}}
	deployment := r.generateDeployment(sr, gatewayModeExternal, testDurableBootstrap)
	if len(deployment.Spec.Template.Spec.TopologySpreadConstraints) != 1 {
		t.Fatalf("topology spread = %#v", deployment.Spec.Template.Spec.TopologySpreadConstraints)
	}
	container := deployment.Spec.Template.Spec.Containers[0]
	requireBackendDispatchContainerPort(t, container, testDurableBootstrap.BackendDispatchPort)
	requireHTTPSManagementReadiness(t, container.ReadinessProbe)
	if deployment.Spec.Template.Annotations["vllm.ai/bootstrap-revision"] != testDurableBootstrap.Revision {
		t.Fatalf("bootstrap revision annotation = %#v", deployment.Spec.Template.Annotations)
	}
}

func requireBackendDispatchContainerPort(t *testing.T, container corev1.Container, wantPort int32) {
	t.Helper()
	for _, port := range container.Ports {
		if port.ContainerPort != wantPort {
			continue
		}
		if port.Name != backendDispatchPortName {
			t.Fatalf("backend dispatch container port name = %q", port.Name)
		}
		if len(port.Name) > 15 {
			t.Fatalf("backend dispatch container port name %q exceeds 15 characters", port.Name)
		}
		return
	}
	t.Fatalf("backend dispatch port missing from %#v", container.Ports)
}

func requireHTTPSManagementReadiness(t *testing.T, probe *corev1.Probe) {
	t.Helper()
	if probe == nil {
		t.Fatal("Management readiness probe is missing")
	}
	if probe.HTTPGet == nil {
		t.Fatalf("Management readiness probe = %#v", probe)
	}
	if probe.HTTPGet.Scheme != corev1.URISchemeHTTPS {
		t.Fatalf("Management readiness scheme = %q", probe.HTTPGet.Scheme)
	}
	if probe.HTTPGet.Path != "/ready" {
		t.Fatalf("Management readiness path = %q", probe.HTTPGet.Path)
	}
	if probe.HTTPGet.Port.StrVal != "management" {
		t.Fatalf("Management readiness port = %#v", probe.HTTPGet.Port)
	}
}

func TestStoreOnlyDeploymentUsesPlaintextOperationalReadiness(t *testing.T) {
	r := &SemanticRouterReconciler{}
	sr := &vllmv1alpha1.SemanticRouter{ObjectMeta: metav1.ObjectMeta{Name: "router"}}
	storeOnly := bootstrapDeploymentContract{
		ManagementStore:     true,
		ManagementPort:      9080,
		BackendDispatchPort: 8180,
	}
	deployment := r.generateDeployment(sr, gatewayModeExternal, storeOnly)
	container := deployment.Spec.Template.Spec.Containers[0]
	listenerPortFound := false
	for _, port := range container.Ports {
		if port.Name == "management" && port.ContainerPort == storeOnly.ManagementPort {
			listenerPortFound = true
		}
	}
	if !listenerPortFound {
		t.Fatalf("store-only operational listener missing from %#v", container.Ports)
	}
	if container.ReadinessProbe == nil || container.ReadinessProbe.HTTPGet == nil {
		t.Fatalf("operational readiness probe = %#v", container.ReadinessProbe)
	}
	if container.ReadinessProbe.HTTPGet.Scheme != corev1.URISchemeHTTP ||
		container.ReadinessProbe.HTTPGet.Path != "/ready" ||
		container.ReadinessProbe.HTTPGet.Port.StrVal != "management" {
		t.Fatalf("store-only readiness HTTPGet = %#v", container.ReadinessProbe.HTTPGet)
	}

	services := r.generateServices(sr, gatewayModeExternal, storeOnly)
	for _, service := range services {
		if service.Name == "router-management" {
			t.Fatalf("store-only deployment exposed a Management API Service: %#v", service)
		}
	}
}

func TestDurableNetworkPolicySeparatesListenerPeers(t *testing.T) {
	inferencePeers := []networkingv1.NetworkPolicyPeer{{
		PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"role": "gateway"}},
	}}
	managementPeers := []networkingv1.NetworkPolicyPeer{{
		PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"role": "console"}},
	}}
	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"},
		Spec: vllmv1alpha1.SemanticRouterSpec{NetworkPolicy: vllmv1alpha1.NetworkPolicySpec{
			InferencePeers: inferencePeers, ManagementPeers: managementPeers,
		}},
	}
	policy := generateNetworkPolicy(sr, gatewayModeExternal, testDurableBootstrap)
	if len(policy.Spec.Ingress) != 3 {
		t.Fatalf("durable ingress rules = %#v", policy.Spec.Ingress)
	}
	if got := policy.Spec.Ingress[0].Ports[0].Port.IntVal; got != DefaultGRPCPort {
		t.Fatalf("inference policy port = %d, want %d", got, DefaultGRPCPort)
	}
	if got := policy.Spec.Ingress[1].Ports[0].Port.IntVal; got != testDurableBootstrap.ManagementPort {
		t.Fatalf("Management policy port = %d, want %d", got, testDurableBootstrap.ManagementPort)
	}
	if got := policy.Spec.Ingress[2].Ports[0].Port.IntVal; got != testDurableBootstrap.BackendDispatchPort {
		t.Fatalf("dispatch policy port = %d, want %d", got, testDurableBootstrap.BackendDispatchPort)
	}
}

func TestDurableNetworkPolicyDefaultsToFailClosedListeners(t *testing.T) {
	sr := &vllmv1alpha1.SemanticRouter{ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"}}
	policy := generateNetworkPolicy(sr, gatewayModeExternal, testDurableBootstrap)
	if len(policy.Spec.Ingress) != 1 {
		t.Fatalf("durable default ingress rules = %#v, want only internal dispatch", policy.Spec.Ingress)
	}
	rule := policy.Spec.Ingress[0]
	if len(rule.From) != 1 || rule.From[0].PodSelector == nil ||
		!reflect.DeepEqual(rule.From[0].PodSelector.MatchLabels, semanticRouterLabels(sr)) {
		t.Fatalf("durable dispatch peer = %#v", rule.From)
	}
	if len(rule.Ports) != 1 || rule.Ports[0].Port.IntVal != testDurableBootstrap.BackendDispatchPort {
		t.Fatalf("durable default ingress ports = %#v", rule.Ports)
	}
}

func TestManagementMigrationGatesRouterWorkload(t *testing.T) {
	r, sr := newManagementMigrationGateFixture(t)
	requirePendingManagementMigration(t, r, sr)
	completeManagementMigration(t, r, sr)
	requireSucceededManagementMigration(t, r, sr)
}

func newManagementMigrationGateFixture(
	t *testing.T,
) (*SemanticRouterReconciler, *vllmv1alpha1.SemanticRouter) {
	t.Helper()
	routerScheme := runtime.NewScheme()
	if err := scheme.AddToScheme(routerScheme); err != nil {
		t.Fatal(err)
	}
	if err := vllmv1alpha1.AddToScheme(routerScheme); err != nil {
		t.Fatal(err)
	}
	immutable := true
	bootstrapConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "router-bootstrap-v1", Namespace: "default"},
		Immutable:  &immutable,
		Data: map[string]string{"config.yaml": `version: v0.3
global:
  stores:
    management:
      postgres:
        dsn_env: TEST_DATABASE_URL
    runtime:
      redis:
        url_env: TEST_REDIS_URL
  services:
    management_api:
      enabled: true
      port: 9443
    access:
      enabled: true
    backend_dispatch:
      port: 8181
`},
	}
	disabled := false
	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default", UID: types.UID("router-uid")},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Bootstrap: vllmv1alpha1.BootstrapSpec{ConfigMapRef: vllmv1alpha1.BootstrapConfigMapReference{
				Name: bootstrapConfigMap.Name, Key: "config.yaml",
			}},
			Persistence:    vllmv1alpha1.PersistenceSpec{Enabled: &disabled},
			ServiceAccount: vllmv1alpha1.ServiceAccountSpec{Create: &disabled},
			Env: []corev1.EnvVar{
				{
					Name: "TEST_DATABASE_URL",
					ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{
						LocalObjectReference: corev1.LocalObjectReference{Name: "management-store"},
						Key:                  "dsn",
					}},
				},
				{Name: "TEST_REDIS_URL", Value: "redacted-for-test"},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(routerScheme).WithObjects(sr, bootstrapConfigMap).Build()
	return &SemanticRouterReconciler{Client: cl, Scheme: routerScheme}, sr
}

func requirePendingManagementMigration(
	t *testing.T,
	r *SemanticRouterReconciler,
	sr *vllmv1alpha1.SemanticRouter,
) {
	t.Helper()
	result, err := r.reconcileOwnedResources(context.Background(), sr, logr.Discard())
	if err != nil {
		t.Fatalf("first reconcileOwnedResources() error = %v", err)
	}
	if result.RequeueAfter == 0 {
		t.Fatalf("migration gate result = %#v, status = %#v", result, sr.Status.Migration)
	}
	if sr.Status.Migration == nil || sr.Status.Migration.State != migrationStatePending {
		t.Fatalf("migration gate status = %#v", sr.Status.Migration)
	}
	deployment := &appsv1.Deployment{}
	if err := r.Get(context.Background(), types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, deployment); err == nil {
		t.Fatal("Router Deployment was created before schema migration completed")
	}
}

func completeManagementMigration(
	t *testing.T,
	r *SemanticRouterReconciler,
	sr *vllmv1alpha1.SemanticRouter,
) {
	t.Helper()
	job := &batchv1.Job{}
	if err := r.Get(context.Background(), types.NamespacedName{Name: sr.Status.Migration.JobName, Namespace: sr.Namespace}, job); err != nil {
		t.Fatalf("migration Job not found: %v", err)
	}
	job.Status.Conditions = []batchv1.JobCondition{{Type: batchv1.JobComplete, Status: corev1.ConditionTrue}}
	if err := r.Status().Update(context.Background(), job); err != nil {
		t.Fatalf("complete migration Job: %v", err)
	}
}

func requireSucceededManagementMigration(
	t *testing.T,
	r *SemanticRouterReconciler,
	sr *vllmv1alpha1.SemanticRouter,
) {
	t.Helper()
	result, err := r.reconcileOwnedResources(context.Background(), sr, logr.Discard())
	if err != nil {
		t.Fatalf("post-migration reconcileOwnedResources() error = %v", err)
	}
	if result.RequeueAfter != 0 {
		t.Fatalf("post-migration result = %#v, status = %#v", result, sr.Status.Migration)
	}
	if sr.Status.Migration == nil || sr.Status.Migration.State != migrationStateSucceeded {
		t.Fatalf("post-migration status = %#v", sr.Status.Migration)
	}

	deployment := &appsv1.Deployment{}
	if err := r.Get(context.Background(), types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, deployment); err != nil {
		t.Fatalf("Router Deployment not created after schema migration: %v", err)
	}
	managementService := &corev1.Service{}
	if err := r.Get(context.Background(), types.NamespacedName{Name: sr.Name + "-management", Namespace: sr.Namespace}, managementService); err != nil {
		t.Fatalf("private Management Service not created: %v", err)
	}
	if !managementService.Spec.PublishNotReadyAddresses {
		t.Fatal("reconciled Management Service must publish bootstrap endpoints before inference readiness")
	}
	requireReadinessGatedServices(t, r, sr)
}

func requireReadinessGatedServices(
	t *testing.T,
	r *SemanticRouterReconciler,
	sr *vllmv1alpha1.SemanticRouter,
) {
	t.Helper()
	for _, name := range []string{sr.Name, sr.Name + "-backend-dispatch", sr.Name + "-metrics"} {
		service := &corev1.Service{}
		if err := r.Get(context.Background(), types.NamespacedName{Name: name, Namespace: sr.Namespace}, service); err != nil {
			t.Fatalf("reconciled readiness-gated Service %s not found: %v", name, err)
		}
		if service.Spec.PublishNotReadyAddresses {
			t.Fatalf("reconciled Service %s must remain readiness-gated", name)
		}
	}
}
