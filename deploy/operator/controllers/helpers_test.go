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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func TestGetInt32OrDefault(t *testing.T) {
	r := &SemanticRouterReconciler{}

	tests := []struct {
		name string
		val  *int32
		def  int32
		want int32
	}{
		{
			name: "nil value returns default",
			val:  nil,
			def:  10,
			want: 10,
		},
		{
			name: "non-nil value returns value",
			val:  func() *int32 { i := int32(5); return &i }(),
			def:  10,
			want: 5,
		},
		{
			name: "zero value returns zero",
			val:  func() *int32 { i := int32(0); return &i }(),
			def:  10,
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := r.getInt32OrDefault(tt.val, tt.def)
			if got != tt.want {
				t.Errorf("getInt32OrDefault() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConvertStringValue(t *testing.T) {
	r := &SemanticRouterReconciler{}

	tests := []struct {
		name  string
		input string
		want  interface{}
	}{
		{
			name:  "integer string",
			input: "100",
			want:  int64(100),
		},
		{
			name:  "float string",
			input: "0.6",
			want:  float64(0.6),
		},
		{
			name:  "boolean true",
			input: "true",
			want:  true,
		},
		{
			name:  "boolean false",
			input: "false",
			want:  false,
		},
		{
			name:  "regular string",
			input: "hello",
			want:  "hello",
		},
		{
			name:  "empty string",
			input: "",
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := r.convertStringValue(tt.input)
			if got != tt.want {
				t.Errorf("convertStringValue(%q) = %v (type %T), want %v (type %T)",
					tt.input, got, got, tt.want, tt.want)
			}
		})
	}
}

func TestConvertStringsToFloats(t *testing.T) {
	r := &SemanticRouterReconciler{}

	tests := []struct {
		name  string
		input map[string]interface{}
		want  map[string]interface{}
	}{
		{
			name: "convert float strings",
			input: map[string]interface{}{
				"threshold": "0.6",
				"name":      "test",
				"count":     "100",
			},
			want: map[string]interface{}{
				"threshold": float64(0.6),
				"name":      "test",
				"count":     int64(100),
			},
		},
		{
			name: "nested maps",
			input: map[string]interface{}{
				"config": map[string]interface{}{
					"threshold": "0.8",
					"enabled":   "true",
				},
			},
			want: map[string]interface{}{
				"config": map[string]interface{}{
					"threshold": float64(0.8),
					"enabled":   true,
				},
			},
		},
		{
			name: "arrays",
			input: map[string]interface{}{
				"values": []interface{}{"0.1", "0.2", "100"},
			},
			want: map[string]interface{}{
				"values": []interface{}{float64(0.1), float64(0.2), int64(100)},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r.convertStringsToFloats(tt.input)
			// Deep comparison would be complex, just check a few key conversions
			for k, expectedVal := range tt.want {
				actualVal, ok := tt.input[k]
				if !ok {
					t.Errorf("key %q not found in result", k)
					continue
				}

				// Type assertion to compare
				switch expected := expectedVal.(type) {
				case float64:
					actual, ok := actualVal.(float64)
					if !ok || actual != expected {
						t.Errorf("key %q: got %v (type %T), want %v (type %T)",
							k, actualVal, actualVal, expectedVal, expectedVal)
					}
				case int64:
					actual, ok := actualVal.(int64)
					if !ok || actual != expected {
						t.Errorf("key %q: got %v (type %T), want %v (type %T)",
							k, actualVal, actualVal, expectedVal, expectedVal)
					}
				case bool:
					actual, ok := actualVal.(bool)
					if !ok || actual != expected {
						t.Errorf("key %q: got %v (type %T), want %v (type %T)",
							k, actualVal, actualVal, expectedVal, expectedVal)
					}
				}
			}
		})
	}
}

func TestValidateSemanticCacheConfig(t *testing.T) {
	tests := []struct {
		name        string
		cache       *vllmv1alpha1.SemanticCacheConfig
		expectError bool
		errorMsg    string
	}{
		{
			name:        "nil cache config is valid",
			cache:       nil,
			expectError: false,
		},
		{
			name: "disabled cache is valid",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled: false,
			},
			expectError: false,
		},
		{
			name: "memory backend is valid without additional config",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "memory",
			},
			expectError: false,
		},
		{
			name: "empty backend type defaults to memory",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "",
			},
			expectError: false,
		},
		{
			name: "redis without config fails",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "redis",
				Redis:       nil,
			},
			expectError: true,
			errorMsg:    "redis configuration required",
		},
		{
			name: "redis without host fails",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "redis",
				Redis: &vllmv1alpha1.RedisCacheConfig{
					Connection: vllmv1alpha1.RedisCacheConnection{
						Host: "",
					},
				},
			},
			expectError: true,
			errorMsg:    "redis.connection.host is required",
		},
		{
			name: "redis with valid config succeeds",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "redis",
				Redis: &vllmv1alpha1.RedisCacheConfig{
					Connection: vllmv1alpha1.RedisCacheConnection{
						Host: "redis.default.svc",
						Port: 6379,
					},
				},
			},
			expectError: false,
		},
		{
			name: "milvus without config fails",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "milvus",
				Milvus:      nil,
			},
			expectError: true,
			errorMsg:    "milvus configuration required",
		},
		{
			name: "milvus without host fails",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "milvus",
				Milvus: &vllmv1alpha1.MilvusCacheConfig{
					Connection: vllmv1alpha1.MilvusCacheConnection{
						Host: "",
					},
				},
			},
			expectError: true,
			errorMsg:    "milvus.connection.host is required",
		},
		{
			name: "milvus with valid config succeeds",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "milvus",
				Milvus: &vllmv1alpha1.MilvusCacheConfig{
					Connection: vllmv1alpha1.MilvusCacheConnection{
						Host: "milvus.default.svc",
						Port: 19530,
					},
				},
			},
			expectError: false,
		},
		{
			name: "hybrid without milvus config fails",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "hybrid",
				Milvus:      nil,
			},
			expectError: true,
			errorMsg:    "milvus configuration required for hybrid",
		},
		{
			name: "hybrid with valid config succeeds",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "hybrid",
				Milvus: &vllmv1alpha1.MilvusCacheConfig{
					Connection: vllmv1alpha1.MilvusCacheConnection{
						Host: "milvus.default.svc",
					},
				},
				HNSW: &vllmv1alpha1.HNSWCacheConfig{
					UseHNSW: true,
					M:       16,
				},
			},
			expectError: false,
		},
		{
			name: "unsupported backend type fails",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "invalid",
			},
			expectError: true,
			errorMsg:    "unsupported backend_type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateSemanticCacheConfig(tt.cache)
			if tt.expectError {
				if err == nil {
					t.Errorf("validateSemanticCacheConfig() expected error containing %q, got nil", tt.errorMsg)
				} else if tt.errorMsg != "" && !contains(err.Error(), tt.errorMsg) {
					t.Errorf("validateSemanticCacheConfig() error = %q, want error containing %q", err.Error(), tt.errorMsg)
				}
			} else {
				if err != nil {
					t.Errorf("validateSemanticCacheConfig() unexpected error = %v", err)
				}
			}
		})
	}
}

func TestGetSecretValue(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = vllmv1alpha1.AddToScheme(scheme)

	tests := []struct {
		name        string
		secret      *corev1.Secret
		selector    *corev1.SecretKeySelector
		expectError bool
		expectedVal string
	}{
		{
			name: "successfully retrieve secret value",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-secret",
					Namespace: "default",
				},
				Data: map[string][]byte{
					"password": []byte("mysecretpassword"),
				},
			},
			selector: &corev1.SecretKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: "test-secret",
				},
				Key: "password",
			},
			expectError: false,
			expectedVal: "mysecretpassword",
		},
		{
			name:        "nil selector returns error",
			secret:      nil,
			selector:    nil,
			expectError: true,
		},
		{
			name: "secret not found returns error",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "wrong-secret",
					Namespace: "default",
				},
				Data: map[string][]byte{
					"password": []byte("test"),
				},
			},
			selector: &corev1.SecretKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: "test-secret",
				},
				Key: "password",
			},
			expectError: true,
		},
		{
			name: "key not found in secret returns error",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-secret",
					Namespace: "default",
				},
				Data: map[string][]byte{
					"other-key": []byte("value"),
				},
			},
			selector: &corev1.SecretKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: "test-secret",
				},
				Key: "password",
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objs []runtime.Object
			if tt.secret != nil {
				objs = append(objs, tt.secret)
			}

			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithRuntimeObjects(objs...).
				Build()

			r := &SemanticRouterReconciler{
				Client: fakeClient,
				Scheme: scheme,
			}

			val, err := r.getSecretValue(context.Background(), "default", tt.selector)

			if tt.expectError {
				if err == nil {
					t.Errorf("getSecretValue() expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("getSecretValue() unexpected error = %v", err)
				}
				if val != tt.expectedVal {
					t.Errorf("getSecretValue() = %q, want %q", val, tt.expectedVal)
				}
			}
		})
	}
}

func TestResolveSemanticCacheSecrets(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = vllmv1alpha1.AddToScheme(scheme)

	redisSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "redis-credentials",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"password": []byte("redis-password"),
		},
	}

	milvusSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "milvus-credentials",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"password": []byte("milvus-password"),
		},
	}

	tests := []struct {
		name        string
		sr          *vllmv1alpha1.SemanticRouter
		secrets     []runtime.Object
		expectError bool
		checkFunc   func(*testing.T, *vllmv1alpha1.SemanticRouter)
	}{
		{
			name: "nil cache config does nothing",
			sr: &vllmv1alpha1.SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-router",
					Namespace: "default",
				},
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Config: vllmv1alpha1.ConfigSpec{
						SemanticCache: nil,
					},
				},
			},
			secrets:     nil,
			expectError: false,
		},
		{
			name: "resolve redis password from secret",
			sr: &vllmv1alpha1.SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-router",
					Namespace: "default",
				},
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Config: vllmv1alpha1.ConfigSpec{
						SemanticCache: &vllmv1alpha1.SemanticCacheConfig{
							Enabled:     true,
							BackendType: "redis",
							Redis: &vllmv1alpha1.RedisCacheConfig{
								Connection: vllmv1alpha1.RedisCacheConnection{
									Host: "redis.svc",
									PasswordSecretRef: &corev1.SecretKeySelector{
										LocalObjectReference: corev1.LocalObjectReference{
											Name: "redis-credentials",
										},
										Key: "password",
									},
								},
							},
						},
					},
				},
			},
			secrets:     []runtime.Object{redisSecret},
			expectError: false,
			checkFunc: func(t *testing.T, sr *vllmv1alpha1.SemanticRouter) {
				if sr.Spec.Config.SemanticCache.Redis.Connection.Password != "redis-password" {
					t.Errorf("Redis password not resolved, got %q", sr.Spec.Config.SemanticCache.Redis.Connection.Password)
				}
				if sr.Spec.Config.SemanticCache.Redis.Connection.PasswordSecretRef != nil {
					t.Errorf("PasswordSecretRef should be cleared after resolution")
				}
			},
		},
		{
			name: "resolve milvus password from secret",
			sr: &vllmv1alpha1.SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-router",
					Namespace: "default",
				},
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Config: vllmv1alpha1.ConfigSpec{
						SemanticCache: &vllmv1alpha1.SemanticCacheConfig{
							Enabled:     true,
							BackendType: "milvus",
							Milvus: &vllmv1alpha1.MilvusCacheConfig{
								Connection: vllmv1alpha1.MilvusCacheConnection{
									Host: "milvus.svc",
									Auth: vllmv1alpha1.MilvusCacheAuth{
										Enabled:  true,
										Username: "root",
										PasswordSecretRef: &corev1.SecretKeySelector{
											LocalObjectReference: corev1.LocalObjectReference{
												Name: "milvus-credentials",
											},
											Key: "password",
										},
									},
								},
							},
						},
					},
				},
			},
			secrets:     []runtime.Object{milvusSecret},
			expectError: false,
			checkFunc: func(t *testing.T, sr *vllmv1alpha1.SemanticRouter) {
				if sr.Spec.Config.SemanticCache.Milvus.Connection.Auth.Password != "milvus-password" {
					t.Errorf("Milvus password not resolved, got %q", sr.Spec.Config.SemanticCache.Milvus.Connection.Auth.Password)
				}
				if sr.Spec.Config.SemanticCache.Milvus.Connection.Auth.PasswordSecretRef != nil {
					t.Errorf("PasswordSecretRef should be cleared after resolution")
				}
			},
		},
		{
			name: "error if redis secret not found",
			sr: &vllmv1alpha1.SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-router",
					Namespace: "default",
				},
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Config: vllmv1alpha1.ConfigSpec{
						SemanticCache: &vllmv1alpha1.SemanticCacheConfig{
							Enabled:     true,
							BackendType: "redis",
							Redis: &vllmv1alpha1.RedisCacheConfig{
								Connection: vllmv1alpha1.RedisCacheConnection{
									PasswordSecretRef: &corev1.SecretKeySelector{
										LocalObjectReference: corev1.LocalObjectReference{
											Name: "missing-secret",
										},
										Key: "password",
									},
								},
							},
						},
					},
				},
			},
			secrets:     nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithRuntimeObjects(tt.secrets...).
				Build()

			r := &SemanticRouterReconciler{
				Client: fakeClient,
				Scheme: scheme,
			}

			err := r.resolveSemanticCacheSecrets(context.Background(), tt.sr)

			if tt.expectError {
				if err == nil {
					t.Errorf("resolveSemanticCacheSecrets() expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("resolveSemanticCacheSecrets() unexpected error = %v", err)
				}
				if tt.checkFunc != nil {
					tt.checkFunc(t, tt.sr)
				}
			}
		})
	}
}

func TestConvertToConfigMapWithRedis(t *testing.T) {
	r := &SemanticRouterReconciler{}

	cache := &vllmv1alpha1.SemanticCacheConfig{
		Enabled:             true,
		BackendType:         "redis",
		SimilarityThreshold: "0.85",
		TTLSeconds:          3600,
		Redis: &vllmv1alpha1.RedisCacheConfig{
			Connection: vllmv1alpha1.RedisCacheConnection{
				Host:     "redis.default.svc",
				Port:     6379,
				Database: 0,
				Password: "test-password",
				Timeout:  30,
			},
			Index: vllmv1alpha1.RedisCacheIndex{
				Name:   "test_idx",
				Prefix: "cache:",
				VectorField: vllmv1alpha1.RedisCacheVectorField{
					Name:       "embedding",
					Dimension:  384,
					MetricType: "COSINE",
				},
				IndexType: "HNSW",
				Params: vllmv1alpha1.RedisCacheIndexParams{
					M:              16,
					EfConstruction: 64,
				},
			},
		},
	}

	result := r.convertToConfigMap(cache)
	configMap, ok := result.(map[string]interface{})
	if !ok {
		t.Fatalf("convertToConfigMap() did not return map[string]interface{}, got %T", result)
	}

	// Verify top-level fields
	if configMap["enabled"] != true {
		t.Errorf("enabled = %v, want true", configMap["enabled"])
	}
	if configMap["backend_type"] != "redis" {
		t.Errorf("backend_type = %v, want redis", configMap["backend_type"])
	}
	if configMap["similarity_threshold"] != float64(0.85) {
		t.Errorf("similarity_threshold = %v (type %T), want 0.85 (float64)", configMap["similarity_threshold"], configMap["similarity_threshold"])
	}
	if configMap["ttl_seconds"] != float64(3600) {
		t.Errorf("ttl_seconds = %v, want 3600", configMap["ttl_seconds"])
	}

	// Verify Redis nested structure
	redis, ok := configMap["redis"].(map[string]interface{})
	if !ok {
		t.Fatalf("redis field is not a map, got %T", configMap["redis"])
	}

	connection, ok := redis["connection"].(map[string]interface{})
	if !ok {
		t.Fatalf("redis.connection is not a map, got %T", redis["connection"])
	}

	if connection["host"] != "redis.default.svc" {
		t.Errorf("redis.connection.host = %v, want redis.default.svc", connection["host"])
	}
	if connection["port"] != float64(6379) {
		t.Errorf("redis.connection.port = %v (type %T), want 6379", connection["port"], connection["port"])
	}

	index, ok := redis["index"].(map[string]interface{})
	if !ok {
		t.Fatalf("redis.index is not a map, got %T", redis["index"])
	}

	if index["name"] != "test_idx" {
		t.Errorf("redis.index.name = %v, want test_idx", index["name"])
	}

	vectorField, ok := index["vector_field"].(map[string]interface{})
	if !ok {
		t.Fatalf("redis.index.vector_field is not a map, got %T", index["vector_field"])
	}

	if vectorField["dimension"] != float64(384) {
		t.Errorf("redis.index.vector_field.dimension = %v, want 384", vectorField["dimension"])
	}
}

func TestConvertToConfigMapWithMilvus(t *testing.T) {
	r := &SemanticRouterReconciler{}

	cache := &vllmv1alpha1.SemanticCacheConfig{
		Enabled:             true,
		BackendType:         "milvus",
		SimilarityThreshold: "0.90",
		TTLSeconds:          7200,
		EmbeddingModel:      "qwen3",
		Milvus: &vllmv1alpha1.MilvusCacheConfig{
			Connection: vllmv1alpha1.MilvusCacheConnection{
				Host:     "milvus.default.svc",
				Port:     19530,
				Database: "semantic_cache",
				Timeout:  30,
				Auth: vllmv1alpha1.MilvusCacheAuth{
					Enabled:  true,
					Username: "root",
					Password: "test-password",
				},
			},
			Collection: vllmv1alpha1.MilvusCacheCollection{
				Name:        "cache",
				Description: "Test cache",
				VectorField: vllmv1alpha1.MilvusCacheVectorField{
					Name:       "embedding",
					Dimension:  1024,
					MetricType: "IP",
				},
				Index: vllmv1alpha1.MilvusCacheCollectionIndex{
					Type: "HNSW",
					Params: vllmv1alpha1.MilvusCacheIndexParams{
						M:              16,
						EfConstruction: 64,
					},
				},
			},
			Search: vllmv1alpha1.MilvusCacheSearch{
				Params: vllmv1alpha1.MilvusCacheSearchParams{
					Ef: 64,
				},
				TopK:             10,
				ConsistencyLevel: "Session",
			},
		},
	}

	result := r.convertToConfigMap(cache)
	configMap, ok := result.(map[string]interface{})
	if !ok {
		t.Fatalf("convertToConfigMap() did not return map[string]interface{}, got %T", result)
	}

	// Verify top-level fields
	if configMap["backend_type"] != "milvus" {
		t.Errorf("backend_type = %v, want milvus", configMap["backend_type"])
	}
	if configMap["embedding_model"] != "qwen3" {
		t.Errorf("embedding_model = %v, want qwen3", configMap["embedding_model"])
	}

	// Verify Milvus nested structure
	milvus, ok := configMap["milvus"].(map[string]interface{})
	if !ok {
		t.Fatalf("milvus field is not a map, got %T", configMap["milvus"])
	}

	connection, ok := milvus["connection"].(map[string]interface{})
	if !ok {
		t.Fatalf("milvus.connection is not a map, got %T", milvus["connection"])
	}

	if connection["host"] != "milvus.default.svc" {
		t.Errorf("milvus.connection.host = %v, want milvus.default.svc", connection["host"])
	}
	if connection["port"] != float64(19530) {
		t.Errorf("milvus.connection.port = %v, want 19530", connection["port"])
	}

	auth, ok := connection["auth"].(map[string]interface{})
	if !ok {
		t.Fatalf("milvus.connection.auth is not a map, got %T", connection["auth"])
	}

	if auth["enabled"] != true {
		t.Errorf("milvus.connection.auth.enabled = %v, want true", auth["enabled"])
	}
	if auth["username"] != "root" {
		t.Errorf("milvus.connection.auth.username = %v, want root", auth["username"])
	}

	collection, ok := milvus["collection"].(map[string]interface{})
	if !ok {
		t.Fatalf("milvus.collection is not a map, got %T", milvus["collection"])
	}

	vectorField, ok := collection["vector_field"].(map[string]interface{})
	if !ok {
		t.Fatalf("milvus.collection.vector_field is not a map, got %T", collection["vector_field"])
	}

	if vectorField["dimension"] != float64(1024) {
		t.Errorf("milvus.collection.vector_field.dimension = %v, want 1024", vectorField["dimension"])
	}

	search, ok := milvus["search"].(map[string]interface{})
	if !ok {
		t.Fatalf("milvus.search is not a map, got %T", milvus["search"])
	}

	if search["consistency_level"] != "Session" {
		t.Errorf("milvus.search.consistency_level = %v, want Session", search["consistency_level"])
	}
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && stringContains(s, substr)))
}

func stringContains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
