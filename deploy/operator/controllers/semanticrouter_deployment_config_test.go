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
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func deploymentConfigTestRouter(t *testing.T) (*SemanticRouterReconciler, *vllmv1alpha1.SemanticRouter) {
	t.Helper()
	s := runtime.NewScheme()
	if err := scheme.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	if err := vllmv1alpha1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			PodAnnotations: map[string]string{"example.com/owner": "user"},
			VLLMEndpoints: []vllmv1alpha1.VLLMEndpointSpec{{
				Name: "backend", Model: "local/model",
				Backend: vllmv1alpha1.VLLMBackend{Type: "service", Service: &vllmv1alpha1.ServiceBackend{Name: "model-old", Port: 8000}},
			}},
		},
	}
	return &SemanticRouterReconciler{Client: fake.NewClientBuilder().WithScheme(s).WithObjects(sr).Build(), Scheme: s}, sr
}

func reconcileDeploymentConfigTest(t *testing.T, r *SemanticRouterReconciler, sr *vllmv1alpha1.SemanticRouter, mode string) *appsv1.Deployment {
	t.Helper()
	ctx := context.Background()
	if err := r.reconcileConfigMap(ctx, sr); err != nil {
		t.Fatal(err)
	}
	if err := r.reconcileDeployment(ctx, sr, mode); err != nil {
		t.Fatal(err)
	}
	deployment := &appsv1.Deployment{}
	if err := r.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, deployment); err != nil {
		t.Fatal(err)
	}
	return deployment
}

func TestBackendConfigUpdateRollsDeployment(t *testing.T) {
	r, sr := deploymentConfigTestRouter(t)
	before := reconcileDeploymentConfigTest(t, r, sr, GatewayModeStandalone)
	checksum := before.Spec.Template.Annotations[deploymentConfigChecksumAnnotation]
	if checksum == "" || before.Spec.Template.Annotations["example.com/owner"] != "user" {
		t.Fatal("PodTemplate must have a checksum and retain user annotations")
	}
	if _, exists := sr.Spec.PodAnnotations[deploymentConfigChecksumAnnotation]; exists {
		t.Fatal("reconciliation mutated user-owned CR annotations")
	}
	unchanged := reconcileDeploymentConfigTest(t, r, sr, GatewayModeStandalone)
	if !reflect.DeepEqual(before.Spec.Template, unchanged.Spec.Template) || before.ResourceVersion != unchanged.ResourceVersion {
		t.Fatal("identical reconciliation must not trigger another rollout")
	}

	sr.Spec.VLLMEndpoints[0].Backend.Service.Name = "model-new"
	sr.Spec.VLLMEndpoints[0].Backend.Service.Port = 9000
	after := reconcileDeploymentConfigTest(t, r, sr, GatewayModeStandalone)
	if checksum == after.Spec.Template.Annotations[deploymentConfigChecksumAnnotation] {
		t.Fatal("changing the discovered backend must roll the Router pod")
	}
	for _, name := range []string{sr.Name + "-config"} {
		cm := &corev1.ConfigMap{}
		if err := r.Get(context.Background(), types.NamespacedName{Name: name, Namespace: sr.Namespace}, cm); err != nil {
			t.Fatal(err)
		}
		for key, data := range cm.Data {
			if key == "tools_db.json" {
				continue
			}
			if !strings.Contains(data, "model-new.default.svc.cluster.local") || strings.Contains(data, "model-old") {
				t.Fatalf("%s/%s does not contain the updated discovered backend", name, key)
			}
		}
	}
}

func TestDeploymentChecksumTracksMountedConfigContent(t *testing.T) {
	for _, mode := range []string{GatewayModeStandalone, GatewayModeIntegration} {
		t.Run(mode, func(t *testing.T) {
			r, sr := deploymentConfigTestRouter(t)
			before := reconcileDeploymentConfigTest(t, r, sr, mode)
			for _, suffix := range []string{"-config"} {
				cm := &corev1.ConfigMap{}
				key := types.NamespacedName{Name: sr.Name + suffix, Namespace: sr.Namespace}
				if err := r.Get(context.Background(), key, cm); err != nil {
					t.Fatal(err)
				}
				cm.Annotations = map[string]string{"example.com/note": "metadata only"}
				if err := r.Update(context.Background(), cm); err != nil {
					t.Fatal(err)
				}
				check := r.generateDeployment(sr, mode)
				if err := r.annotateDeploymentConfig(context.Background(), sr, check); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(before.Spec.Template, check.Spec.Template) {
					t.Fatal("ConfigMap metadata changes must not restart pods")
				}
				cm.Data["revision.txt"] = "new content"
				if err := r.Update(context.Background(), cm); err != nil {
					t.Fatal(err)
				}
				if err := r.annotateDeploymentConfig(context.Background(), sr, check); err != nil {
					t.Fatal(err)
				}
				if reflect.DeepEqual(before.Spec.Template, check.Spec.Template) {
					t.Fatalf("%s content changes must restart pods", suffix)
				}
				before = check
			}
		})
	}
}

func TestDeploymentWaitsForMountedConfiguration(t *testing.T) {
	r, sr := deploymentConfigTestRouter(t)
	if err := r.reconcileDeployment(context.Background(), sr, GatewayModeStandalone); err == nil {
		t.Fatal("missing mounted configuration must not produce an unversioned deployment")
	}
}

func TestRetiredEnvoyConfigIsDeletedOnceTheRolloutCompletes(t *testing.T) {
	r, sr := deploymentConfigTestRouter(t)
	ctx := context.Background()
	retired := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: sr.Name + retiredEnvoyConfigSuffix, Namespace: sr.Namespace}}
	if err := controllerutil.SetControllerReference(sr, retired, r.Scheme); err != nil {
		t.Fatal(err)
	}
	foreign := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "other" + retiredEnvoyConfigSuffix, Namespace: sr.Namespace}}
	for _, cm := range []*corev1.ConfigMap{retired, foreign} {
		if err := r.Create(ctx, cm); err != nil {
			t.Fatal(err)
		}
	}
	deployment := reconcileDeploymentConfigTest(t, r, sr, GatewayModeStandalone)

	deployment.Status = appsv1.DeploymentStatus{ObservedGeneration: deployment.Generation, Replicas: 2, UpdatedReplicas: 1}
	if err := r.Status().Update(ctx, deployment); err != nil {
		t.Fatal(err)
	}
	if err := r.deleteRetiredEnvoyConfig(ctx, sr); err != nil {
		t.Fatal(err)
	}
	if err := r.Get(ctx, client.ObjectKeyFromObject(retired), &corev1.ConfigMap{}); err != nil {
		t.Fatalf("a Pod of the previous rollout may still mount the retired ConfigMap: %v", err)
	}

	deployment.Status = appsv1.DeploymentStatus{ObservedGeneration: deployment.Generation, Replicas: 1, UpdatedReplicas: 1}
	if err := r.Status().Update(ctx, deployment); err != nil {
		t.Fatal(err)
	}
	if err := r.deleteRetiredEnvoyConfig(ctx, sr); err != nil {
		t.Fatal(err)
	}
	if err := r.Get(ctx, client.ObjectKeyFromObject(retired), &corev1.ConfigMap{}); !apierrors.IsNotFound(err) {
		t.Fatalf("retired Envoy ConfigMap survived the completed rollout: %v", err)
	}
	if err := r.Get(ctx, client.ObjectKeyFromObject(foreign), &corev1.ConfigMap{}); err != nil {
		t.Fatalf("a ConfigMap the SemanticRouter doesn't control must stay: %v", err)
	}
}
