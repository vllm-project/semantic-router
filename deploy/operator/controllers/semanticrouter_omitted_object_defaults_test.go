package controllers

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	corev1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/defaulting"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/yaml"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

// The API server applies a property default only inside an object the CR
// wrote. A full Update used to write serviceAccount and persistence back as
// empty objects, which is what gave a CR that omitted them create=true and
// enabled=true. A finalizer change is now a metadata patch, so the CRD has to
// default the two objects itself: without that the reconciler reads a nil
// flag, skips the ServiceAccount the pod names, and drops the PVC.
func TestOmittedServiceAccountAndPersistenceAreDefaultedAndReconciled(t *testing.T) {
	shapes := []struct {
		name string
		// spec holds the serviceAccount and persistence the CR wrote, if any.
		spec map[string]interface{}
		want bool
	}{
		{name: "omitted", spec: map[string]interface{}{}, want: true},
		{
			name: "empty object",
			spec: map[string]interface{}{
				"serviceAccount": map[string]interface{}{},
				"persistence":    map[string]interface{}{},
			},
			want: true,
		},
		{
			name: "explicit false",
			spec: map[string]interface{}{
				"serviceAccount": map[string]interface{}{"create": false},
				"persistence":    map[string]interface{}{"enabled": false},
			},
		},
	}
	for _, crdPath := range []string{
		filepath.Join("..", "config", "crd", "bases", "vllm.ai_semanticrouters.yaml"),
		filepath.Join("..", "bundle", "manifests", "vllm.ai_semanticrouters.yaml"),
	} {
		structural := loadSemanticRouterStructural(t, crdPath)
		for _, shape := range shapes {
			t.Run(filepath.Base(filepath.Dir(crdPath))+"/"+shape.name, func(t *testing.T) {
				sr := admitSemanticRouter(t, structural, shape.spec)

				s := runtime.NewScheme()
				_ = scheme.AddToScheme(s)
				_ = vllmv1alpha1.AddToScheme(s)
				// The CRD default for persistence.storageClassName.
				standard := &storagev1.StorageClass{ObjectMeta: metav1.ObjectMeta{Name: "standard"}}
				cl := fake.NewClientBuilder().WithScheme(s).WithObjects(sr, standard).Build()
				r := &SemanticRouterReconciler{Client: cl, Scheme: s}
				ctx := context.Background()

				if err := r.reconcileServiceAccount(ctx, sr); err != nil {
					t.Fatalf("reconcileServiceAccount: %v", err)
				}
				err := cl.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, &corev1.ServiceAccount{})
				if created := err == nil; created != shape.want || (err != nil && !apierrors.IsNotFound(err)) {
					t.Errorf("ServiceAccount created = %t (error %v), want %t", created, err, shape.want)
				}

				if err = r.reconcilePVC(ctx, sr); err != nil {
					t.Fatalf("reconcilePVC: %v", err)
				}
				pvc := &corev1.PersistentVolumeClaim{}
				err = cl.Get(ctx, types.NamespacedName{Name: sr.Name + "-models", Namespace: sr.Namespace}, pvc)
				if created := err == nil; created != shape.want || (err != nil && !apierrors.IsNotFound(err)) {
					t.Fatalf("PVC created = %t (error %v), want %t", created, err, shape.want)
				}
				if !shape.want {
					return
				}
				// The rest of the persistence defaults arrive with the object.
				if modes := pvc.Spec.AccessModes; len(modes) != 1 || modes[0] != corev1.ReadWriteOnce {
					t.Errorf("PVC access modes = %v, want [%s]", modes, corev1.ReadWriteOnce)
				}
				if class := ptr.Deref(pvc.Spec.StorageClassName, ""); class != "standard" {
					t.Errorf("PVC storage class = %q, want standard", class)
				}
				if size := pvc.Spec.Resources.Requests[corev1.ResourceStorage]; size.Cmp(resource.MustParse("10Gi")) != 0 {
					t.Errorf("PVC size = %s, want 10Gi", size.String())
				}
			})
		}
	}
}

// loadSemanticRouterStructural reads a generated CRD copy into the structural
// schema the API server defaults custom resources with.
func loadSemanticRouterStructural(t *testing.T, path string) *schema.Structural {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read CRD: %v", err)
	}
	var crd apiextensionsv1.CustomResourceDefinition
	if err = yaml.Unmarshal(data, &crd); err != nil {
		t.Fatalf("parse CRD: %v", err)
	}
	if len(crd.Spec.Versions) == 0 || crd.Spec.Versions[0].Schema == nil {
		t.Fatal("CRD carries no schema")
	}
	var internal apiextensions.JSONSchemaProps
	if err = apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(crd.Spec.Versions[0].Schema.OpenAPIV3Schema, &internal, nil); err != nil {
		t.Fatalf("convert schema: %v", err)
	}
	structural, err := schema.NewStructural(&internal)
	if err != nil {
		t.Fatalf("structural schema: %v", err)
	}
	return structural
}

// admitSemanticRouter returns the SemanticRouter a reconcile reads after the
// API server has defaulted a CR with this spec.
func admitSemanticRouter(t *testing.T, structural *schema.Structural, spec map[string]interface{}) *vllmv1alpha1.SemanticRouter {
	t.Helper()
	object := map[string]interface{}{
		"apiVersion": "vllm.ai/v1alpha1",
		"kind":       "SemanticRouter",
		"metadata":   map[string]interface{}{"name": "router", "namespace": "default"},
		"spec":       spec,
	}
	defaulting.Default(object, structural)
	data, err := json.Marshal(object)
	if err != nil {
		t.Fatalf("encode defaulted CR: %v", err)
	}
	sr := &vllmv1alpha1.SemanticRouter{}
	if err = json.Unmarshal(data, sr); err != nil {
		t.Fatalf("decode defaulted CR: %v", err)
	}
	return sr
}
