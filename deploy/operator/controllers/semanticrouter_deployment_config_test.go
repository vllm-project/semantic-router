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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

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
	if err := r.reconcileEnvoyConfig(ctx, sr, mode); err != nil {
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
	before := reconcileDeploymentConfigTest(t, r, sr, "standalone")
	checksum := before.Spec.Template.Annotations[deploymentConfigChecksumAnnotation]
	if checksum == "" || before.Spec.Template.Annotations["example.com/owner"] != "user" {
		t.Fatal("PodTemplate must have a checksum and retain user annotations")
	}
	if _, exists := sr.Spec.PodAnnotations[deploymentConfigChecksumAnnotation]; exists {
		t.Fatal("reconciliation mutated user-owned CR annotations")
	}
	unchanged := reconcileDeploymentConfigTest(t, r, sr, "standalone")
	if !reflect.DeepEqual(before.Spec.Template, unchanged.Spec.Template) || before.ResourceVersion != unchanged.ResourceVersion {
		t.Fatal("identical reconciliation must not trigger another rollout")
	}

	sr.Spec.VLLMEndpoints[0].Backend.Service.Name = "model-new"
	sr.Spec.VLLMEndpoints[0].Backend.Service.Port = 9000
	after := reconcileDeploymentConfigTest(t, r, sr, "standalone")
	if checksum == after.Spec.Template.Annotations[deploymentConfigChecksumAnnotation] {
		t.Fatal("changing the discovered backend must roll the Envoy and Router pod")
	}
	for _, name := range []string{sr.Name + "-config", sr.Name + "-envoy-config"} {
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
	for _, mode := range []string{"standalone", "gateway"} {
		t.Run(mode, func(t *testing.T) {
			r, sr := deploymentConfigTestRouter(t)
			before := reconcileDeploymentConfigTest(t, r, sr, mode)
			suffixes := []string{"-config"}
			if mode == "standalone" {
				suffixes = append(suffixes, "-envoy-config")
			}
			for _, suffix := range suffixes {
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
				if err := r.annotateDeploymentConfig(context.Background(), sr, mode, check); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(before.Spec.Template, check.Spec.Template) {
					t.Fatal("ConfigMap metadata changes must not restart pods")
				}
				// Exercise each mounted ConfigMap independently; an Envoy-only
				// bootstrap change must roll even if Router YAML did not change.
				cm.Data["revision.txt"] = "new content"
				if err := r.Update(context.Background(), cm); err != nil {
					t.Fatal(err)
				}
				if err := r.annotateDeploymentConfig(context.Background(), sr, mode, check); err != nil {
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
	if err := r.reconcileDeployment(context.Background(), sr, "standalone"); err == nil {
		t.Fatal("missing mounted configuration must not produce an unversioned deployment")
	}
}
