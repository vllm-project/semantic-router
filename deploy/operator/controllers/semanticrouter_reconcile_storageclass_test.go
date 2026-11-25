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
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

// storageClassReconcileRouter builds a Router that manages its own model claim
// on storageClassName, with that claim already bound to a volume. generatePVC
// names the claim after the Router, so it is "<name>-models".
func storageClassReconcileRouter(t *testing.T, storageClassName string) (*SemanticRouterReconciler, *vllmv1alpha1.SemanticRouter, client.Client) {
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
			Image: vllmv1alpha1.ImageSpec{Repository: "ghcr.io/vllm-project/semantic-router/vllm-sr", Tag: "old"},
			Persistence: vllmv1alpha1.PersistenceSpec{
				Enabled:          ptr.To(true),
				StorageClassName: storageClassName,
			},
		},
	}

	bound := &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "router-models", Namespace: "default"},
		Spec: corev1.PersistentVolumeClaimSpec{
			StorageClassName: ptr.To(storageClassName),
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("10Gi")},
			},
		},
		Status: corev1.PersistentVolumeClaimStatus{
			Phase: corev1.ClaimBound,
		},
	}

	// The class the bound claim was provisioned with. The first test retires
	// it, so it has to exist to be deleted.
	class := &storagev1.StorageClass{
		ObjectMeta:  metav1.ObjectMeta{Name: storageClassName},
		Provisioner: "example.com/driver",
	}

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr, bound, class).Build()
	return &SemanticRouterReconciler{Client: cl, Scheme: s}, sr, cl
}

// routerImage is the tag the Deployment's Router container runs after a full
// reconcile of the claim and the Deployment.
func routerImage(t *testing.T, r *SemanticRouterReconciler, cl client.Client, sr *vllmv1alpha1.SemanticRouter) string {
	t.Helper()
	ctx := context.Background()
	if err := r.reconcileConfigMap(ctx, sr); err != nil {
		t.Fatalf("reconcileConfigMap: %v", err)
	}
	if err := r.reconcileDeployment(ctx, sr, GatewayModeStandalone); err != nil {
		t.Fatalf("reconcileDeployment: %v", err)
	}
	deployment := &appsv1.Deployment{}
	if err := cl.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, deployment); err != nil {
		t.Fatal(err)
	}
	return deployment.Spec.Template.Spec.Containers[0].Image
}

// A StorageClass retirement must not strand the Router on its previous image.
// The claim is already bound, so nothing here creates a claim and the class it
// was bound with is no longer needed: the reconcile that only updates the
// Deployment has to run.
func TestReconcilePVCReusesBoundClaimAfterStorageClassRetired(t *testing.T) {
	r, sr, cl := storageClassReconcileRouter(t, "old-model-class")
	ctx := context.Background()

	if err := r.reconcilePVC(ctx, sr); err != nil {
		t.Fatalf("reconcilePVC with the class present: %v", err)
	}
	if got, want := routerImage(t, r, cl, sr), "ghcr.io/vllm-project/semantic-router/vllm-sr:old"; got != want {
		t.Fatalf("Deployment image = %q, want %q", got, want)
	}

	// Retire the class, keep the claim bound, and move the image forward.
	retired := &storagev1.StorageClass{ObjectMeta: metav1.ObjectMeta{Name: "old-model-class"}}
	if err := cl.Delete(ctx, retired); err != nil {
		t.Fatal(err)
	}
	sr.Spec.Image.Tag = "new"

	if err := r.reconcilePVC(ctx, sr); err != nil {
		t.Fatalf("reconcilePVC after the class was retired: %v", err)
	}
	if got, want := routerImage(t, r, cl, sr), "ghcr.io/vllm-project/semantic-router/vllm-sr:new"; got != want {
		t.Errorf("Deployment image = %q, want %q: a retired StorageClass must not block the update", got, want)
	}

	// The bound claim survives untouched.
	claim := &corev1.PersistentVolumeClaim{}
	if err := cl.Get(ctx, types.NamespacedName{Name: "router-models", Namespace: "default"}, claim); err != nil {
		t.Fatalf("the bound model claim must survive: %v", err)
	}
	if claim.Status.Phase != corev1.ClaimBound {
		t.Errorf("model claim phase = %q, want %q", claim.Status.Phase, corev1.ClaimBound)
	}
}

// A reconcile that has to create the claim still needs the class, and still
// says so when it is gone.
func TestReconcilePVCReportsMissingStorageClassWhenCreatingClaim(t *testing.T) {
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
			Persistence: vllmv1alpha1.PersistenceSpec{
				Enabled:          ptr.To(true),
				StorageClassName: "gone-class",
			},
		},
	}

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr).Build()
	r := &SemanticRouterReconciler{Client: cl, Scheme: s}

	err := r.reconcilePVC(context.Background(), sr)
	if err == nil {
		t.Fatal("reconcilePVC must report a missing StorageClass when it has to create the claim")
	}
	if want := `StorageClass "gone-class" not found`; !strings.Contains(err.Error(), want) {
		t.Errorf("reconcilePVC error = %q, want it to mention %q", err.Error(), want)
	}

	claim := &corev1.PersistentVolumeClaim{}
	if getErr := cl.Get(context.Background(), types.NamespacedName{Name: "router-models", Namespace: "default"}, claim); getErr == nil {
		t.Error("no claim may be created without its StorageClass")
	}
}
