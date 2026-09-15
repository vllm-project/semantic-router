package k8s

import (
	"context"
	"errors"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	kubetesting "k8s.io/client-go/testing"
)

func apiConflictError(name string) error {
	return apierrors.NewConflict(schema.GroupResource{Resource: "configmaps"}, name, errors.New("concurrent update"))
}

func boolPtr(b bool) *bool { return &b }

func newConfigMap(namespace, name string, data map[string]string, owners []metav1.OwnerReference) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       namespace,
			Name:            name,
			OwnerReferences: owners,
		},
		Data: data,
	}
}

func TestConfigMapWriterWrite(t *testing.T) {
	target := ConfigMapTarget{Namespace: "vllm-semantic-router-system", Name: "semantic-router-config", Key: "config.yaml"}
	existing := newConfigMap(target.Namespace, target.Name, map[string]string{
		"config.yaml":   "version: v0.3\n",
		"tools_db.json": "{}",
	}, nil)

	clientset := fakeclientset.NewSimpleClientset(existing)
	writer := NewConfigMapWriter(clientset)

	if err := writer.Write(context.Background(), target, []byte("version: v0.3\nlisteners: []\n")); err != nil {
		t.Fatalf("Write() error = %v, want nil", err)
	}

	updated, err := clientset.CoreV1().ConfigMaps(target.Namespace).Get(context.Background(), target.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get updated ConfigMap: %v", err)
	}
	if updated.Data["config.yaml"] != "version: v0.3\nlisteners: []\n" {
		t.Errorf("config.yaml = %q, want the written document", updated.Data["config.yaml"])
	}
	// A key this package does not own must survive the write untouched.
	if updated.Data["tools_db.json"] != "{}" {
		t.Errorf("tools_db.json = %q, want it preserved", updated.Data["tools_db.json"])
	}
}

func TestConfigMapWriterRefusesControllerOwnedConfigMap(t *testing.T) {
	target := ConfigMapTarget{Namespace: "ns", Name: "operator-managed-config", Key: "config.yaml"}
	owners := []metav1.OwnerReference{{
		APIVersion: "vllm.ai/v1alpha1",
		Kind:       "SemanticRouter",
		Name:       "prod",
		Controller: boolPtr(true),
	}}
	existing := newConfigMap(target.Namespace, target.Name, map[string]string{"config.yaml": "version: v0.3\n"}, owners)

	clientset := fakeclientset.NewSimpleClientset(existing)
	writer := NewConfigMapWriter(clientset)

	err := writer.Write(context.Background(), target, []byte("version: v0.3\nlisteners: []\n"))
	if !errors.Is(err, ErrConfigMapControllerOwned) {
		t.Fatalf("Write() error = %v, want ErrConfigMapControllerOwned", err)
	}

	unchanged, getErr := clientset.CoreV1().ConfigMaps(target.Namespace).Get(context.Background(), target.Name, metav1.GetOptions{})
	if getErr != nil {
		t.Fatalf("get ConfigMap: %v", getErr)
	}
	if unchanged.Data["config.yaml"] != "version: v0.3\n" {
		t.Errorf("controller-owned ConfigMap was modified: %q", unchanged.Data["config.yaml"])
	}
}

func TestConfigMapWriterNonControllerOwnerIsWritable(t *testing.T) {
	// An owner reference exists (e.g. a ReplicaSet-style owner) but does not
	// mark itself as a controller. Only a controller owner reverts edits on
	// reconcile, so this must not be refused.
	target := ConfigMapTarget{Namespace: "ns", Name: "config", Key: "config.yaml"}
	owners := []metav1.OwnerReference{{
		APIVersion: "v1",
		Kind:       "ConfigMap",
		Name:       "some-owner",
		Controller: boolPtr(false),
	}}
	existing := newConfigMap(target.Namespace, target.Name, map[string]string{"config.yaml": "old"}, owners)
	clientset := fakeclientset.NewSimpleClientset(existing)
	writer := NewConfigMapWriter(clientset)

	if err := writer.Write(context.Background(), target, []byte("new")); err != nil {
		t.Fatalf("Write() error = %v, want nil", err)
	}
}

func TestConfigMapWriterMissingConfigMap(t *testing.T) {
	target := ConfigMapTarget{Namespace: "ns", Name: "missing", Key: "config.yaml"}
	clientset := fakeclientset.NewSimpleClientset()
	writer := NewConfigMapWriter(clientset)

	err := writer.Write(context.Background(), target, []byte("version: v0.3\n"))
	if err == nil {
		t.Fatal("Write() = nil, want a not-found error")
	}
}

func TestConfigMapWriterRetriesOnConflict(t *testing.T) {
	target := ConfigMapTarget{Namespace: "ns", Name: "config", Key: "config.yaml"}
	existing := newConfigMap(target.Namespace, target.Name, map[string]string{"config.yaml": "v1"}, nil)
	clientset := fakeclientset.NewSimpleClientset(existing)

	// The first Update call simulates a concurrent write landing between our
	// Get and Update (another replica's request, or a controller reconcile);
	// RetryOnConflict must re-read and try again rather than surface it.
	attempts := 0
	clientset.PrependReactor("update", "configmaps", func(action kubetesting.Action) (bool, runtime.Object, error) {
		attempts++
		if attempts == 1 {
			return true, nil, apiConflictError(target.Name)
		}
		return false, nil, nil
	})

	writer := NewConfigMapWriter(clientset)
	if err := writer.Write(context.Background(), target, []byte("v2")); err != nil {
		t.Fatalf("Write() error = %v, want nil after retry", err)
	}
	if attempts < 2 {
		t.Fatalf("attempts = %d, want at least 2 (the retry never happened)", attempts)
	}
}

func TestConfigMapTargetFromEnv(t *testing.T) {
	t.Setenv(ConfigMapNameEnv, "")
	t.Setenv(ConfigMapNamespaceEnv, "")
	t.Setenv(ConfigMapKeyEnv, "")
	if _, ok := ConfigMapTargetFromEnv(); ok {
		t.Fatal("ConfigMapTargetFromEnv() ok = true with no env set, want false")
	}

	t.Setenv(ConfigMapNameEnv, "semantic-router-config")
	t.Setenv(ConfigMapNamespaceEnv, "vllm-semantic-router-system")
	target, ok := ConfigMapTargetFromEnv()
	if !ok {
		t.Fatal("ConfigMapTargetFromEnv() ok = false, want true")
	}
	if target.Key != "config.yaml" {
		t.Errorf("Key defaulted to %q, want config.yaml", target.Key)
	}

	t.Setenv(ConfigMapKeyEnv, "custom.yaml")
	target, _ = ConfigMapTargetFromEnv()
	if target.Key != "custom.yaml" {
		t.Errorf("Key = %q, want the explicit override", target.Key)
	}
}
