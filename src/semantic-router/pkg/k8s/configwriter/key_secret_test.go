package configwriter

import (
	"context"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestKeySecretStoresEachKeyOnceUnderAnEnvironmentName(t *testing.T) {
	client := fake.NewSimpleClientset(&corev1.Secret{ObjectMeta: metav1.ObjectMeta{Namespace: "router", Name: "router-keys"}})
	store := KeySecret{Client: client, Namespace: "router", Name: "router-keys"}

	if ref, created, err := store.Ref("sk-test-one", false); err != nil || created || ref != (config.KeyRef{}) {
		t.Fatalf("a plan found a key never stored: %+v %v (%v)", ref, created, err)
	}
	first, created, err := store.Ref("sk-test-one", true)
	if err != nil || !created || !regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`).MatchString(first.Env) {
		t.Fatalf("ref = %+v, created %v (%v), want a new environment name", first, created, err)
	}
	if again, againCreated, againErr := store.Ref("sk-test-one", true); againErr != nil || againCreated || again != first {
		t.Fatalf("the stored key reads as %+v, created %v (%v), want %+v", again, againCreated, againErr, first)
	}
	second, _, err := store.Ref("sk-test-two", true)
	if err != nil || second == first {
		t.Fatalf("a second key got %+v (%v)", second, err)
	}
	if err = store.Remove(second); err != nil {
		t.Fatal(err)
	}
	secret, err := client.CoreV1().Secrets("router").Get(context.Background(), "router-keys", metav1.GetOptions{})
	if err != nil || len(secret.Data) != 1 || string(secret.Data[first.Env]) != "sk-test-one" {
		t.Fatalf("Secret data = %v (%v), want the first key only", secret.Data, err)
	}
}

// A Secret that is missing or that the pod may not write (rbac.create=false)
// keeps the key in the ConfigMap, as before the key Secret, with a warning.
func TestAKeySecretThePodCannotWriteKeepsTheKey(t *testing.T) {
	forbidden := fake.NewSimpleClientset(&corev1.Secret{ObjectMeta: metav1.ObjectMeta{Namespace: "router", Name: "router-keys"}})
	forbidden.PrependReactor("get", "secrets", func(k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, apierrors.NewForbidden(schema.GroupResource{Resource: "secrets"}, "router-keys", nil)
	})
	for name, client := range map[string]*fake.Clientset{"missing": fake.NewSimpleClientset(), "forbidden": forbidden} {
		t.Run(name, func(t *testing.T) {
			ref, created, err := KeySecret{Client: client, Namespace: "router", Name: "router-keys"}.Ref("sk-test-one", true)
			if err != nil || created || ref != (config.KeyRef{}) {
				t.Fatalf("ref = %+v, created %v (%v), want the key kept", ref, created, err)
			}
		})
	}
}

func TestKeyStorePicksTheStoreOfTheDeployment(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	t.Setenv(config.KeyFileGroupEnv, strconv.Itoa(42))
	store, err := KeyStore(configPath, config.RouterKeyDir, nil)
	if err != nil || store != (config.KeyDir{Dir: filepath.Join(filepath.Dir(configPath), config.RouterKeyDir), Group: 42}) {
		t.Fatalf("local store = %+v (%v)", store, err)
	}
	t.Setenv(config.KeyFileGroupEnv, "stack")
	if _, err = KeyStore(configPath, config.RouterKeyDir, nil); err == nil {
		t.Fatal("a group that is not a number was accepted")
	}

	t.Setenv(ConfigMapNameEnv, "router-config")
	t.Setenv(ConfigMapNamespaceEnv, "router")
	if store, err = KeyStore(configPath, config.RouterKeyDir, nil); err != nil {
		t.Fatal(err)
	}
	if ref, _, storeErr := store.Ref("sk-test-one", true); storeErr != nil || ref != (config.KeyRef{}) {
		t.Fatalf("a ConfigMap target without a key Secret stored the key: %+v (%v)", ref, storeErr)
	}
	t.Setenv(KeySecretNameEnv, "router-keys")
	client := fake.NewSimpleClientset()
	store, err = KeyStore(configPath, config.RouterKeyDir, func() (*ConfigMapWriter, error) { return NewConfigMapWriter(client), nil })
	if err != nil || store != (KeySecret{Client: client, Namespace: "router", Name: "router-keys"}) {
		t.Fatalf("ConfigMap store = %+v (%v)", store, err)
	}
}
