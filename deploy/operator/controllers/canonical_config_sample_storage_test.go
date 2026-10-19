package controllers

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	kubeyaml "sigs.k8s.io/yaml"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Exercise the real sample -> Secret resolution -> ConfigMap -> router loader
// boundary. Backend connectivity and cache operations belong to storage tests.
func TestStorageSamplesSurviveConfigMapReconciliation(t *testing.T) {
	for _, backend := range []string{"redis", "milvus", "qdrant", "hybrid"} {
		t.Run(backend, func(t *testing.T) {
			r, sr, secret := storageSampleReconciler(t, backend)
			if secret != nil {
				if err := r.Create(context.Background(), secret); err != nil {
					t.Fatal(err)
				}
			}
			if err := r.reconcileConfigMap(context.Background(), sr); err != nil {
				t.Fatal(err)
			}
			cm := &corev1.ConfigMap{}
			if err := r.Get(context.Background(), types.NamespacedName{Name: sr.Name + "-config", Namespace: sr.Namespace}, cm); err != nil {
				t.Fatal(err)
			}
			data := cm.Data["config.yaml"]
			if strings.Contains(data, "password_secret_ref") {
				t.Fatal("router config contains an unresolved Kubernetes Secret reference")
			}
			cfg, err := routerconfig.ParseYAMLBytes([]byte(data))
			if err != nil {
				t.Fatal(err)
			}
			cache := cfg.SemanticCache
			if !cache.Enabled || cache.BackendType != backend {
				t.Fatalf("cache backend lost in translation: %+v", cache)
			}
			switch backend {
			case "redis":
				redis := cache.Redis
				if redis == nil || redis.Connection.Host != "redis.cache-backends.svc.cluster.local" ||
					redis.Connection.Port != 6379 || redis.Connection.Password != "roundtrip-password" ||
					redis.Index.Name != "semantic_cache_idx" || redis.Index.VectorField.Dimension != 1024 ||
					redis.Index.IndexType != "HNSW" || redis.Search.TopK != 1 {
					t.Fatalf("Redis connection, Secret or index settings lost: %+v", redis)
				}
			case "milvus", "hybrid":
				milvus := cache.Milvus
				if milvus == nil || milvus.Connection.Host != "milvus-standalone.cache-backends.svc.cluster.local" ||
					milvus.Connection.Port != 19530 || milvus.Connection.Database != "semantic_router_cache" ||
					!milvus.Connection.Auth.Enabled || milvus.Connection.Auth.Username != "root" ||
					milvus.Connection.Auth.Password != "roundtrip-password" {
					t.Fatalf("Milvus connection or Secret settings lost: %+v", milvus)
				}
				if backend == "milvus" {
					if milvus.Collection.Name != "semantic_cache" || milvus.Collection.VectorField.Dimension != 768 ||
						milvus.Search.TopK != 10 || milvus.Search.ConsistencyLevel != "Session" {
						t.Fatalf("Milvus collection or search settings lost: %+v", milvus)
					}
				} else if milvus.Collection.Name != "semantic_cache_hybrid" || milvus.Collection.VectorField.Dimension != 384 ||
					milvus.Search.TopK != 20 || milvus.Search.ConsistencyLevel != "Eventually" ||
					cache.MaxEntries != 5000 || cache.EvictionPolicy != "lru" {
					t.Fatalf("hybrid local/persistent cache settings lost: %+v", cache)
				}
			case "qdrant":
				qdrant := cache.Qdrant
				if qdrant == nil || qdrant.Host != "qdrant.cache-backends.svc.cluster.local" ||
					qdrant.Port != 6334 || qdrant.CollectionName != "semantic_cache" ||
					qdrant.ConnectTimeout != 10 || qdrant.UseTLS {
					t.Fatalf("Qdrant transport or collection settings lost: %+v", qdrant)
				}
			}
		})
	}
}

func TestStorageSampleReconciliationRejectsUnresolvedSecrets(t *testing.T) {
	for _, backend := range []string{"redis", "milvus", "hybrid"} {
		for _, missing := range []string{"secret", "key"} {
			t.Run(backend+"/missing-"+missing, func(t *testing.T) {
				r, sr, secret := storageSampleReconciler(t, backend)
				if missing == "key" {
					secret.Data = map[string][]byte{"wrong-key": []byte("roundtrip-password")}
					if err := r.Create(context.Background(), secret); err != nil {
						t.Fatal(err)
					}
				}
				if err := r.reconcileConfigMap(context.Background(), sr); err == nil || !strings.Contains(err.Error(), "failed to resolve cache secrets") {
					t.Fatalf("unresolved Secret was not rejected: %v", err)
				}
				configs := &corev1.ConfigMapList{}
				if err := r.List(context.Background(), configs); err != nil {
					t.Fatal(err)
				}
				if len(configs.Items) != 0 {
					t.Fatal("failed Secret resolution wrote a partial ConfigMap")
				}
			})
		}
	}
}

func storageSampleReconciler(t *testing.T, backend string) (*SemanticRouterReconciler, *vllmv1alpha1.SemanticRouter, *corev1.Secret) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "config", "samples", "vllm.ai_v1alpha1_semanticrouter_"+backend+"_cache.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	sr := &vllmv1alpha1.SemanticRouter{}
	if err := kubeyaml.Unmarshal(data, sr); err != nil {
		t.Fatal(err)
	}
	// This direct test isolates cache translation from Kubernetes endpoint discovery.
	sr.Spec.VLLMEndpoints = nil
	s := runtime.NewScheme()
	if err := corev1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	if err := vllmv1alpha1.AddToScheme(s); err != nil {
		t.Fatal(err)
	}
	r := &SemanticRouterReconciler{Client: fake.NewClientBuilder().WithScheme(s).WithObjects(sr).Build(), Scheme: s}
	if backend == "qdrant" {
		return r, sr, nil
	}
	name := "milvus-credentials"
	if backend == "redis" {
		name = "redis-credentials"
	}
	return r, sr, &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: sr.Namespace},
		Data:       map[string][]byte{"password": []byte("roundtrip-password")},
	}
}
