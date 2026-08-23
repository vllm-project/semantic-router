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

var testManagedBootstrap = bootstrapDeploymentContract{
	Mode:                controlPlaneModeManaged,
	Revision:            "sha256:managed-test",
	ManagementPort:      9443,
	BackendDispatchPort: 8181,
	PostgresDSNEnv:      "TEST_DATABASE_URL",
}

func TestManagedMigrationJobUsesExplicitDeploymentInputs(t *testing.T) {
	createServiceAccount := true
	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Image:          vllmv1alpha1.ImageSpec{Repository: "example.invalid/router", Tag: "v0.4.0"},
			ServiceAccount: vllmv1alpha1.ServiceAccountSpec{Create: &createServiceAccount},
			Env: []corev1.EnvVar{{Name: "TEST_DATABASE_URL", ValueFrom: &corev1.EnvVarSource{
				SecretKeyRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "managed-store"}, Key: "dsn"},
			}}},
			EnvFrom: []corev1.EnvFromSource{{SecretRef: &corev1.SecretEnvSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: "router-runtime"},
			}}},
			Volumes:      []corev1.Volume{{Name: "managed-secrets"}},
			VolumeMounts: []corev1.VolumeMount{{Name: "managed-secrets", MountPath: "/run/secrets/router", ReadOnly: true}},
		},
	}

	job, err := (&SemanticRouterReconciler{}).generateMigrationJob(sr, testManagedBootstrap)
	if err != nil {
		t.Fatalf("generateMigrationJob() error = %v", err)
	}
	if job.Spec.Template.Spec.ServiceAccountName != "router" {
		t.Fatalf("migration ServiceAccount = %q", job.Spec.Template.Spec.ServiceAccountName)
	}
	container := job.Spec.Template.Spec.Containers[0]
	if container.Image != "example.invalid/router:v0.4.0" {
		t.Fatalf("migration image = %q", container.Image)
	}
	if want := []string{"--dsn-env", "TEST_DATABASE_URL", "--timeout", "5m"}; !reflect.DeepEqual(container.Args, want) {
		t.Fatalf("migration args = %#v, want %#v", container.Args, want)
	}
	if !reflect.DeepEqual(container.Env, sr.Spec.Env) || !reflect.DeepEqual(container.EnvFrom, sr.Spec.EnvFrom) ||
		!reflect.DeepEqual(container.VolumeMounts, sr.Spec.VolumeMounts) ||
		!reflect.DeepEqual(job.Spec.Template.Spec.Volumes, sr.Spec.Volumes) {
		t.Fatal("migration Job did not inherit the declared Secret projection inputs")
	}

	revised := sr.DeepCopy()
	revised.Spec.Env[0].ValueFrom.SecretKeyRef.Name = "managed-store-v2"
	revisedJob, err := (&SemanticRouterReconciler{}).generateMigrationJob(revised, testManagedBootstrap)
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

func TestManagedServicesSeparateInferenceAndManagement(t *testing.T) {
	r := &SemanticRouterReconciler{}
	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"},
		Spec: vllmv1alpha1.SemanticRouterSpec{Service: vllmv1alpha1.ServiceSpec{
			Management: vllmv1alpha1.ManagementServiceSpec{Port: 8443},
		}},
	}
	services := r.generateServices(sr, gatewayModeExternal, testManagedBootstrap)
	if len(services) != 4 {
		t.Fatalf("managed Service count = %d, want 4", len(services))
	}
	byName := make(map[string]*corev1.Service, len(services))
	for _, service := range services {
		byName[service.Name] = service
	}
	public := byName["router"]
	if public == nil || len(public.Spec.Ports) != 1 || public.Spec.Ports[0].Name != "grpc" {
		t.Fatalf("public inference Service = %#v", public)
	}
	if public.Spec.PublishNotReadyAddresses {
		t.Fatal("public inference Service must remain readiness-gated")
	}
	management := byName["router-management"]
	if management == nil || management.Spec.Type != corev1.ServiceTypeClusterIP ||
		len(management.Spec.Ports) != 1 || management.Spec.Ports[0].Port != 8443 ||
		management.Spec.Ports[0].TargetPort.IntVal != 9443 {
		t.Fatalf("private Management Service = %#v", management)
	}
	if !management.Spec.PublishNotReadyAddresses {
		t.Fatal("private Management Service must be reachable before inference readiness")
	}
	dispatch := byName["router-backend-dispatch"]
	if dispatch == nil || len(dispatch.Spec.Ports) != 1 ||
		dispatch.Spec.Ports[0].Name != backendDispatchPortName {
		t.Fatalf("managed backend dispatch Service = %#v", dispatch)
	}
	if dispatch.Spec.PublishNotReadyAddresses {
		t.Fatal("backend dispatch Service must remain readiness-gated")
	}
	metrics := byName["router-metrics"]
	if metrics == nil {
		t.Fatalf("managed internal Services = %#v", byName)
	}
	if metrics.Spec.PublishNotReadyAddresses {
		t.Fatal("metrics Service must remain readiness-gated")
	}
}

func TestManagedDeploymentUsesHTTPSReadinessAndTopologySpread(t *testing.T) {
	r := &SemanticRouterReconciler{}
	sr := &vllmv1alpha1.SemanticRouter{ObjectMeta: metav1.ObjectMeta{Name: "router"}}
	deployment := r.generateDeployment(sr, gatewayModeExternal, testManagedBootstrap)
	if len(deployment.Spec.Template.Spec.TopologySpreadConstraints) != 1 {
		t.Fatalf("topology spread = %#v", deployment.Spec.Template.Spec.TopologySpreadConstraints)
	}
	container := deployment.Spec.Template.Spec.Containers[0]
	dispatchPortFound := false
	for _, port := range container.Ports {
		if port.ContainerPort != testManagedBootstrap.BackendDispatchPort {
			continue
		}
		dispatchPortFound = true
		if port.Name != backendDispatchPortName || len(port.Name) > 15 {
			t.Fatalf("managed backend dispatch container port = %#v", port)
		}
	}
	if !dispatchPortFound {
		t.Fatalf("managed backend dispatch port missing from %#v", container.Ports)
	}
	if container.ReadinessProbe == nil || container.ReadinessProbe.HTTPGet == nil {
		t.Fatalf("managed readiness probe = %#v", container.ReadinessProbe)
	}
	if container.ReadinessProbe.HTTPGet.Scheme != corev1.URISchemeHTTPS ||
		container.ReadinessProbe.HTTPGet.Path != "/ready" ||
		container.ReadinessProbe.HTTPGet.Port.StrVal != "management" {
		t.Fatalf("managed readiness HTTPGet = %#v", container.ReadinessProbe.HTTPGet)
	}
	if deployment.Spec.Template.Annotations["vllm.ai/bootstrap-revision"] != testManagedBootstrap.Revision {
		t.Fatalf("bootstrap revision annotation = %#v", deployment.Spec.Template.Annotations)
	}
}

func TestManagedNetworkPolicySeparatesListenerPeers(t *testing.T) {
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
	policy := generateNetworkPolicy(sr, gatewayModeExternal, testManagedBootstrap)
	if len(policy.Spec.Ingress) != 3 {
		t.Fatalf("managed ingress rules = %#v", policy.Spec.Ingress)
	}
	if got := policy.Spec.Ingress[0].Ports[0].Port.IntVal; got != DefaultGRPCPort {
		t.Fatalf("inference policy port = %d, want %d", got, DefaultGRPCPort)
	}
	if got := policy.Spec.Ingress[1].Ports[0].Port.IntVal; got != testManagedBootstrap.ManagementPort {
		t.Fatalf("Management policy port = %d, want %d", got, testManagedBootstrap.ManagementPort)
	}
	if got := policy.Spec.Ingress[2].Ports[0].Port.IntVal; got != testManagedBootstrap.BackendDispatchPort {
		t.Fatalf("dispatch policy port = %d, want %d", got, testManagedBootstrap.BackendDispatchPort)
	}
}

func TestManagedNetworkPolicyDefaultsToFailClosedListeners(t *testing.T) {
	sr := &vllmv1alpha1.SemanticRouter{ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"}}
	policy := generateNetworkPolicy(sr, gatewayModeExternal, testManagedBootstrap)
	if len(policy.Spec.Ingress) != 1 {
		t.Fatalf("managed default ingress rules = %#v, want only internal dispatch", policy.Spec.Ingress)
	}
	rule := policy.Spec.Ingress[0]
	if len(rule.From) != 1 || rule.From[0].PodSelector == nil ||
		!reflect.DeepEqual(rule.From[0].PodSelector.MatchLabels, semanticRouterLabels(sr)) {
		t.Fatalf("managed dispatch peer = %#v", rule.From)
	}
	if len(rule.Ports) != 1 || rule.Ports[0].Port.IntVal != testManagedBootstrap.BackendDispatchPort {
		t.Fatalf("managed default ingress ports = %#v", rule.Ports)
	}
}

func TestManagedMigrationGatesRouterWorkload(t *testing.T) {
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
		Data: map[string]string{"config.yaml": `version: v0.4
global:
  control_plane:
    mode: managed
  stores:
    access:
      postgres:
        dsn_env: TEST_DATABASE_URL
  services:
    management_api:
      port: 9443
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
			Env:            []corev1.EnvVar{{Name: "TEST_DATABASE_URL", Value: "redacted-for-test"}},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(routerScheme).WithObjects(sr, bootstrapConfigMap).Build()
	r := &SemanticRouterReconciler{Client: cl, Scheme: routerScheme}

	result, err := r.reconcileOwnedResources(context.Background(), sr, logr.Discard())
	if err != nil {
		t.Fatalf("first reconcileOwnedResources() error = %v", err)
	}
	if result.RequeueAfter == 0 || sr.Status.Migration == nil || sr.Status.Migration.State != migrationStatePending {
		t.Fatalf("migration gate result = %#v, status = %#v", result, sr.Status.Migration)
	}
	deployment := &appsv1.Deployment{}
	if err := cl.Get(context.Background(), types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, deployment); err == nil {
		t.Fatal("Router Deployment was created before schema migration completed")
	}

	job := &batchv1.Job{}
	if err := cl.Get(context.Background(), types.NamespacedName{Name: sr.Status.Migration.JobName, Namespace: sr.Namespace}, job); err != nil {
		t.Fatalf("migration Job not found: %v", err)
	}
	job.Status.Conditions = []batchv1.JobCondition{{Type: batchv1.JobComplete, Status: corev1.ConditionTrue}}
	if err := cl.Status().Update(context.Background(), job); err != nil {
		t.Fatalf("complete migration Job: %v", err)
	}

	result, err = r.reconcileOwnedResources(context.Background(), sr, logr.Discard())
	if err != nil {
		t.Fatalf("post-migration reconcileOwnedResources() error = %v", err)
	}
	if result.RequeueAfter != 0 || sr.Status.Migration.State != migrationStateSucceeded {
		t.Fatalf("post-migration result = %#v, status = %#v", result, sr.Status.Migration)
	}
	if err := cl.Get(context.Background(), types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, deployment); err != nil {
		t.Fatalf("Router Deployment not created after schema migration: %v", err)
	}
	managementService := &corev1.Service{}
	if err := cl.Get(context.Background(), types.NamespacedName{Name: sr.Name + "-management", Namespace: sr.Namespace}, managementService); err != nil {
		t.Fatalf("private Management Service not created: %v", err)
	}
	if !managementService.Spec.PublishNotReadyAddresses {
		t.Fatal("reconciled Management Service must publish bootstrap endpoints before inference readiness")
	}
	for _, name := range []string{sr.Name, sr.Name + "-backend-dispatch", sr.Name + "-metrics"} {
		service := &corev1.Service{}
		if err := cl.Get(context.Background(), types.NamespacedName{Name: name, Namespace: sr.Namespace}, service); err != nil {
			t.Fatalf("reconciled readiness-gated Service %s not found: %v", name, err)
		}
		if service.Spec.PublishNotReadyAddresses {
			t.Fatalf("reconciled Service %s must remain readiness-gated", name)
		}
	}
}
