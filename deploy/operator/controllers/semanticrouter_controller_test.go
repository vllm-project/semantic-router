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
	"testing"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func TestReconcileSemanticRouter(t *testing.T) {
	// Register the vllm scheme
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	// Create a SemanticRouter object
	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-router",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Replicas: func() *int32 { i := int32(2); return &i }(),
		},
	}

	// Create a fake client
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr).WithStatusSubresource(sr).Build()

	// Create the reconciler
	r := &SemanticRouterReconciler{
		Client: cl,
		Scheme: s,
	}

	// Test reconciliation
	req := reconcile.Request{
		NamespacedName: types.NamespacedName{
			Name:      "test-router",
			Namespace: "default",
		},
	}

	// First reconcile should add finalizer
	_, err := r.Reconcile(context.Background(), req)
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}

	// Verify finalizer was added
	updatedSR := &vllmv1alpha1.SemanticRouter{}
	err = cl.Get(context.Background(), req.NamespacedName, updatedSR)
	if err != nil {
		t.Fatalf("failed to get updated SemanticRouter: %v", err)
	}

	found := false
	for _, f := range updatedSR.Finalizers {
		if f == SemanticRouterFinalizer {
			found = true
			break
		}
	}
	if !found {
		t.Error("finalizer was not added")
	}
}

func TestGeneratePVC(t *testing.T) {
	s := runtime.NewScheme()
	_ = vllmv1alpha1.AddToScheme(s)

	r := &SemanticRouterReconciler{
		Scheme: s,
	}

	tests := []struct {
		name    string
		sr      *vllmv1alpha1.SemanticRouter
		wantErr bool
	}{
		{
			name: "valid size",
			sr: &vllmv1alpha1.SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Persistence: vllmv1alpha1.PersistenceSpec{
						Size: "10Gi",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "invalid size",
			sr: &vllmv1alpha1.SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Persistence: vllmv1alpha1.PersistenceSpec{
						Size: "invalid",
					},
				},
			},
			wantErr: true,
		},
		{
			name: "empty size uses default",
			sr: &vllmv1alpha1.SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "default",
				},
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Persistence: vllmv1alpha1.PersistenceSpec{
						Size: "",
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pvc, err := r.generatePVC(tt.sr)
			if (err != nil) != tt.wantErr {
				t.Errorf("generatePVC() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && pvc == nil {
				t.Error("generatePVC() returned nil PVC")
			}
		})
	}
}

func TestGenerateDeployment(t *testing.T) {
	s := runtime.NewScheme()
	_ = vllmv1alpha1.AddToScheme(s)

	r := &SemanticRouterReconciler{
		Scheme: s,
	}

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Replicas: func() *int32 { i := int32(3); return &i }(),
		},
	}

	deployment := r.generateDeployment(sr, "standalone")

	if deployment == nil {
		t.Fatal("generateDeployment() returned nil")
	}

	if *deployment.Spec.Replicas != 3 {
		t.Errorf("expected 3 replicas, got %d", *deployment.Spec.Replicas)
	}

	if deployment.Name != "test" {
		t.Errorf("expected name 'test', got '%s'", deployment.Name)
	}

	if deployment.Namespace != "default" {
		t.Errorf("expected namespace 'default', got '%s'", deployment.Namespace)
	}
}

func TestGetPodSecurityContext(t *testing.T) {
	s := runtime.NewScheme()
	_ = vllmv1alpha1.AddToScheme(s)

	tests := []struct {
		name        string
		isOpenShift bool
		sr          *vllmv1alpha1.SemanticRouter
		wantUser    bool
	}{
		{
			name:        "OpenShift platform",
			isOpenShift: true,
			sr: &vllmv1alpha1.SemanticRouter{
				Spec: vllmv1alpha1.SemanticRouterSpec{},
			},
			wantUser: false, // OpenShift should not set runAsUser
		},
		{
			name:        "Standard Kubernetes",
			isOpenShift: false,
			sr: &vllmv1alpha1.SemanticRouter{
				Spec: vllmv1alpha1.SemanticRouterSpec{},
			},
			wantUser: true, // Standard K8s should set runAsUser
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &SemanticRouterReconciler{
				Scheme:      s,
				isOpenShift: &tt.isOpenShift,
			}

			psc := r.getPodSecurityContext(tt.sr)
			if psc == nil {
				t.Fatal("getPodSecurityContext() returned nil")
			}

			hasUser := psc.RunAsUser != nil
			if hasUser != tt.wantUser {
				t.Errorf("expected runAsUser set=%v, got=%v", tt.wantUser, hasUser)
			}

			if tt.wantUser && *psc.RunAsUser != DefaultRunAsUser {
				t.Errorf("expected runAsUser=%d, got=%d", DefaultRunAsUser, *psc.RunAsUser)
			}
		})
	}
}

func TestUpdateStatus(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{},
	}

	// Test when deployment doesn't exist
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr).WithStatusSubresource(sr).Build()
	r := &SemanticRouterReconciler{
		Client: cl,
		Scheme: s,
	}

	err := r.updateStatus(context.Background(), sr)
	if err != nil {
		t.Fatalf("updateStatus() failed when deployment not found: %v", err)
	}

	// Refetch to check status was updated
	updatedSR := &vllmv1alpha1.SemanticRouter{}
	err = cl.Get(context.Background(), types.NamespacedName{Name: "test", Namespace: "default"}, updatedSR)
	if err != nil {
		t.Fatalf("failed to get updated SemanticRouter: %v", err)
	}

	if updatedSR.Status.Phase != "Pending" {
		t.Errorf("expected phase 'Pending', got '%s'", updatedSR.Status.Phase)
	}

	// Test when deployment exists
	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Status: appsv1.DeploymentStatus{
			Replicas:      2,
			ReadyReplicas: 2,
		},
	}

	cl = fake.NewClientBuilder().WithScheme(s).WithObjects(sr, deployment).WithStatusSubresource(sr).Build()
	r = &SemanticRouterReconciler{
		Client: cl,
		Scheme: s,
	}

	err = r.updateStatus(context.Background(), sr)
	if err != nil {
		t.Fatalf("updateStatus() failed: %v", err)
	}

	// Refetch to check status was updated
	updatedSR2 := &vllmv1alpha1.SemanticRouter{}
	err = cl.Get(context.Background(), types.NamespacedName{Name: "test", Namespace: "default"}, updatedSR2)
	if err != nil {
		t.Fatalf("failed to get updated SemanticRouter: %v", err)
	}

	if updatedSR2.Status.Phase != "Running" {
		t.Errorf("expected phase 'Running', got '%s'", updatedSR2.Status.Phase)
	}

	if updatedSR2.Status.ReadyReplicas != 2 {
		t.Errorf("expected 2 ready replicas, got %d", updatedSR2.Status.ReadyReplicas)
	}
}

func TestFinalizeSemanticRouter(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Persistence: vllmv1alpha1.PersistenceSpec{
				Enabled: func() *bool { b := true; return &b }(),
			},
		},
	}

	pvc := &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-models",
			Namespace: "default",
		},
	}

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr, pvc).Build()
	r := &SemanticRouterReconciler{
		Client: cl,
		Scheme: s,
	}

	err := r.finalizeSemanticRouter(context.Background(), sr)
	if err != nil {
		t.Fatalf("finalizeSemanticRouter() failed: %v", err)
	}

	// Verify PVC was deleted
	foundPVC := &corev1.PersistentVolumeClaim{}
	err = cl.Get(context.Background(), types.NamespacedName{Name: "test-models", Namespace: "default"}, foundPVC)
	if err == nil {
		t.Error("expected PVC to be deleted, but it still exists")
	}
}

func TestParseQuantity(t *testing.T) {
	r := &SemanticRouterReconciler{}

	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{"valid Gi", "10Gi", false},
		{"valid Mi", "500Mi", false},
		{"valid G", "5G", false},
		{"invalid", "invalid", true},
		{"empty", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := r.parseQuantity(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseQuantity(%q) error = %v, wantErr %v", tt.input, err, tt.wantErr)
			}
		})
	}
}

func TestGenerateService(t *testing.T) {
	s := runtime.NewScheme()
	_ = vllmv1alpha1.AddToScheme(s)

	r := &SemanticRouterReconciler{
		Scheme: s,
	}

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{},
	}

	svc := r.generateService(sr, "gateway-integration")

	if svc == nil {
		t.Fatal("generateService() returned nil")
	}

	if svc.Name != "test" {
		t.Errorf("expected name 'test', got '%s'", svc.Name)
	}

	if svc.Namespace != "default" {
		t.Errorf("expected namespace 'default', got '%s'", svc.Namespace)
	}

	if len(svc.Spec.Ports) != 3 {
		t.Errorf("expected 3 ports, got %d", len(svc.Spec.Ports))
	}
}

func TestGenerateHPA(t *testing.T) {
	s := runtime.NewScheme()
	_ = vllmv1alpha1.AddToScheme(s)

	r := &SemanticRouterReconciler{
		Scheme: s,
	}

	minReplicas := int32(2)
	maxReplicas := int32(10)
	cpuTarget := int32(80)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Autoscaling: vllmv1alpha1.AutoscalingSpec{
				MinReplicas:                    &minReplicas,
				MaxReplicas:                    &maxReplicas,
				TargetCPUUtilizationPercentage: &cpuTarget,
			},
		},
	}

	hpa := r.generateHPA(sr)

	if hpa == nil {
		t.Fatal("generateHPA() returned nil")
	}

	if hpa.Name != "test" {
		t.Errorf("expected name 'test', got '%s'", hpa.Name)
	}

	if *hpa.Spec.MinReplicas != 2 {
		t.Errorf("expected minReplicas 2, got %d", *hpa.Spec.MinReplicas)
	}

	if hpa.Spec.MaxReplicas != 10 {
		t.Errorf("expected maxReplicas 10, got %d", hpa.Spec.MaxReplicas)
	}
}

func TestGenerateIngress(t *testing.T) {
	s := runtime.NewScheme()
	_ = vllmv1alpha1.AddToScheme(s)

	r := &SemanticRouterReconciler{
		Scheme: s,
	}

	pathType := string(networkingv1.PathTypePrefix)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Ingress: vllmv1alpha1.IngressSpec{
				Hosts: []vllmv1alpha1.IngressHost{
					{
						Host: "example.com",
						Paths: []vllmv1alpha1.IngressPath{
							{
								Path:     "/",
								PathType: pathType,
							},
						},
					},
				},
			},
		},
	}

	ing := r.generateIngress(sr)

	if ing == nil {
		t.Fatal("generateIngress() returned nil")
	}

	if ing.Name != "test" {
		t.Errorf("expected name 'test', got '%s'", ing.Name)
	}

	if len(ing.Spec.Rules) != 1 {
		t.Errorf("expected 1 rule, got %d", len(ing.Spec.Rules))
	}

	if ing.Spec.Rules[0].Host != "example.com" {
		t.Errorf("expected host 'example.com', got '%s'", ing.Spec.Rules[0].Host)
	}
}

func TestGenerateContainers(t *testing.T) {
	s := runtime.NewScheme()
	_ = vllmv1alpha1.AddToScheme(s)

	r := &SemanticRouterReconciler{
		Scheme: s,
	}

	tests := []struct {
		name          string
		sr            *vllmv1alpha1.SemanticRouter
		expectedImage string
	}{
		{
			name: "default image",
			sr: &vllmv1alpha1.SemanticRouter{
				Spec: vllmv1alpha1.SemanticRouterSpec{},
			},
			expectedImage: DefaultImage,
		},
		{
			name: "custom image",
			sr: &vllmv1alpha1.SemanticRouter{
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Image: vllmv1alpha1.ImageSpec{
						Repository: "custom-image",
						Tag:        "v1.0",
					},
				},
			},
			expectedImage: "custom-image:v1.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			containers := r.generateContainers(tt.sr, "gateway-integration")

			if len(containers) != 1 {
				t.Fatalf("expected 1 container, got %d", len(containers))
			}

			if containers[0].Image != tt.expectedImage {
				t.Errorf("expected image '%s', got '%s'", tt.expectedImage, containers[0].Image)
			}

			if containers[0].Name != "semantic-router" {
				t.Errorf("expected container name 'semantic-router', got '%s'", containers[0].Name)
			}
		})
	}
}

func TestGenerateVolumes(t *testing.T) {
	s := runtime.NewScheme()
	_ = vllmv1alpha1.AddToScheme(s)

	r := &SemanticRouterReconciler{
		Scheme: s,
	}

	tests := []struct {
		name           string
		sr             *vllmv1alpha1.SemanticRouter
		expectedVolume int
	}{
		{
			name: "config and cache volumes",
			sr: &vllmv1alpha1.SemanticRouter{
				Spec: vllmv1alpha1.SemanticRouterSpec{},
			},
			expectedVolume: 3, // config-volume + cache-volume + models-volume (emptyDir)
		},
		{
			name: "config, cache and PVC volumes",
			sr: &vllmv1alpha1.SemanticRouter{
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Persistence: vllmv1alpha1.PersistenceSpec{
						Enabled: func() *bool { b := true; return &b }(),
					},
				},
			},
			expectedVolume: 3, // config-volume + cache-volume + models-volume
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			volumes := r.generateVolumes(tt.sr, "gateway-integration")

			if len(volumes) != tt.expectedVolume {
				t.Errorf("expected %d volumes, got %d", tt.expectedVolume, len(volumes))
			}

			// All configs should have config-volume
			hasConfig := false
			for _, vol := range volumes {
				if vol.Name == "config-volume" {
					hasConfig = true
					break
				}
			}
			if !hasConfig {
				t.Error("expected config-volume not found")
			}
		})
	}
}

func TestReconcileService(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{},
	}

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr).Build()
	r := &SemanticRouterReconciler{
		Client: cl,
		Scheme: s,
	}

	// Test service creation
	err := r.reconcileService(context.Background(), sr, "gateway-integration")
	if err != nil {
		t.Fatalf("reconcileService() failed: %v", err)
	}

	// Verify service was created
	svc := &corev1.Service{}
	err = cl.Get(context.Background(), types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, svc)
	if err != nil {
		t.Fatalf("service not found: %v", err)
	}

	if len(svc.Spec.Ports) != 3 {
		t.Errorf("expected 3 ports, got %d", len(svc.Spec.Ports))
	}
}

func TestReconcileHPA(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	minReplicas := int32(2)
	maxReplicas := int32(10)
	cpuTarget := int32(80)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Autoscaling: vllmv1alpha1.AutoscalingSpec{
				Enabled:                        func() *bool { b := true; return &b }(),
				MinReplicas:                    &minReplicas,
				MaxReplicas:                    &maxReplicas,
				TargetCPUUtilizationPercentage: &cpuTarget,
			},
		},
	}

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr).Build()
	r := &SemanticRouterReconciler{
		Client: cl,
		Scheme: s,
	}

	// Test HPA creation
	err := r.reconcileHPA(context.Background(), sr)
	if err != nil {
		t.Fatalf("reconcileHPA() failed: %v", err)
	}

	// Verify HPA was created
	hpa := &autoscalingv2.HorizontalPodAutoscaler{}
	err = cl.Get(context.Background(), types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, hpa)
	if err != nil {
		t.Fatalf("HPA not found: %v", err)
	}

	if *hpa.Spec.MinReplicas != 2 {
		t.Errorf("expected minReplicas 2, got %d", *hpa.Spec.MinReplicas)
	}

	if hpa.Spec.MaxReplicas != 10 {
		t.Errorf("expected maxReplicas 10, got %d", hpa.Spec.MaxReplicas)
	}
}

func TestReconcileIngress(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	pathType := string(networkingv1.PathTypePrefix)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			Ingress: vllmv1alpha1.IngressSpec{
				Enabled: func() *bool { b := true; return &b }(),
				Hosts: []vllmv1alpha1.IngressHost{
					{
						Host: "example.com",
						Paths: []vllmv1alpha1.IngressPath{
							{
								Path:     "/",
								PathType: pathType,
							},
						},
					},
				},
			},
		},
	}

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr).Build()
	r := &SemanticRouterReconciler{
		Client: cl,
		Scheme: s,
	}

	// Test ingress creation
	err := r.reconcileIngress(context.Background(), sr)
	if err != nil {
		t.Fatalf("reconcileIngress() failed: %v", err)
	}

	// Verify ingress was created
	ing := &networkingv1.Ingress{}
	err = cl.Get(context.Background(), types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, ing)
	if err != nil {
		t.Fatalf("ingress not found: %v", err)
	}

	if len(ing.Spec.Rules) != 1 {
		t.Errorf("expected 1 rule, got %d", len(ing.Spec.Rules))
	}

	if ing.Spec.Rules[0].Host != "example.com" {
		t.Errorf("expected host 'example.com', got '%s'", ing.Spec.Rules[0].Host)
	}
}

func TestReconcileServiceAccount(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{
			ServiceAccount: vllmv1alpha1.ServiceAccountSpec{
				Create: func() *bool { b := true; return &b }(),
			},
		},
	}

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr).Build()
	r := &SemanticRouterReconciler{
		Client: cl,
		Scheme: s,
	}

	// Test service account creation
	err := r.reconcileServiceAccount(context.Background(), sr)
	if err != nil {
		t.Fatalf("reconcileServiceAccount() failed: %v", err)
	}

	// Verify service account was created
	sa := &corev1.ServiceAccount{}
	err = cl.Get(context.Background(), types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, sa)
	if err != nil {
		t.Fatalf("service account not found: %v", err)
	}

	if sa.Name != "test" {
		t.Errorf("expected name 'test', got '%s'", sa.Name)
	}
}

func TestReconcileConfigMap(t *testing.T) {
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = vllmv1alpha1.AddToScheme(s)

	sr := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			Namespace: "default",
		},
		Spec: vllmv1alpha1.SemanticRouterSpec{},
	}

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr).Build()
	r := &SemanticRouterReconciler{
		Client: cl,
		Scheme: s,
	}

	// Test configmap creation
	err := r.reconcileConfigMap(context.Background(), sr)
	if err != nil {
		t.Fatalf("reconcileConfigMap() failed: %v", err)
	}

	// Verify configmap was created
	cm := &corev1.ConfigMap{}
	err = cl.Get(context.Background(), types.NamespacedName{Name: sr.Name + "-config", Namespace: sr.Namespace}, cm)
	if err != nil {
		t.Fatalf("configmap not found: %v", err)
	}

	if _, ok := cm.Data["config.yaml"]; !ok {
		t.Error("expected config.yaml key in configmap")
	}

	if _, ok := cm.Data["tools_db.json"]; !ok {
		t.Error("expected tools_db.json key in configmap")
	}
}

func TestGetContainerSecurityContext(t *testing.T) {
	s := runtime.NewScheme()
	_ = vllmv1alpha1.AddToScheme(s)

	tests := []struct {
		name        string
		isOpenShift bool
		sr          *vllmv1alpha1.SemanticRouter
		wantUser    bool
	}{
		{
			name:        "OpenShift platform",
			isOpenShift: true,
			sr:          &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{}},
			wantUser:    false,
		},
		{
			name:        "Standard Kubernetes",
			isOpenShift: false,
			sr:          &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{}},
			wantUser:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &SemanticRouterReconciler{
				Scheme:      s,
				isOpenShift: &tt.isOpenShift,
			}

			sc := r.getContainerSecurityContext(tt.sr)
			if sc == nil {
				t.Fatal("getContainerSecurityContext() returned nil")
			}

			hasUser := sc.RunAsUser != nil
			if hasUser != tt.wantUser {
				t.Errorf("expected runAsUser set=%v, got=%v", tt.wantUser, hasUser)
			}

			// All platforms should drop ALL capabilities
			if sc.Capabilities == nil || len(sc.Capabilities.Drop) == 0 {
				t.Error("expected capabilities to drop ALL")
			}

			// All platforms should not allow privilege escalation
			if sc.AllowPrivilegeEscalation == nil || *sc.AllowPrivilegeEscalation {
				t.Error("expected allowPrivilegeEscalation to be false")
			}
		})
	}
}
