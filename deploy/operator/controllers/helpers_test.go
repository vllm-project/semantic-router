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
			name: "valkey without config fails",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "valkey",
				Valkey:      nil,
			},
			expectError: true,
			errorMsg:    "valkey configuration required",
		},
		{
			name: "valkey without host fails",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "valkey",
				Valkey: &vllmv1alpha1.ValkeyCacheConfig{
					Connection: vllmv1alpha1.ValkeyCacheConnection{
						Host: "",
					},
				},
			},
			expectError: true,
			errorMsg:    "valkey.connection.host is required",
		},
		{
			name: "valkey with valid config succeeds",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "valkey",
				Valkey: &vllmv1alpha1.ValkeyCacheConfig{
					Connection: vllmv1alpha1.ValkeyCacheConnection{
						Host: "valkey.default.svc",
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
			name: "qdrant without config fails",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "qdrant",
				Qdrant:      nil,
			},
			expectError: true,
			errorMsg:    "qdrant configuration required",
		},
		{
			name: "qdrant without host fails",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "qdrant",
				Qdrant: &vllmv1alpha1.QdrantCacheConfig{
					Host: "",
					Port: 6334,
				},
			},
			expectError: true,
			errorMsg:    "qdrant.host is required",
		},
		{
			name: "qdrant with valid config succeeds",
			cache: &vllmv1alpha1.SemanticCacheConfig{
				Enabled:     true,
				BackendType: "qdrant",
				Qdrant: &vllmv1alpha1.QdrantCacheConfig{
					Host: "qdrant.cache-backends.svc.cluster.local",
					Port: 6334,
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

	valkeySecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valkey-credentials",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"password": []byte("valkey-password"),
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
			name: "resolve valkey password from secret",
			sr: &vllmv1alpha1.SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-router",
					Namespace: "default",
				},
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Config: vllmv1alpha1.ConfigSpec{
						SemanticCache: &vllmv1alpha1.SemanticCacheConfig{
							Enabled:     true,
							BackendType: "valkey",
							Valkey: &vllmv1alpha1.ValkeyCacheConfig{
								Connection: vllmv1alpha1.ValkeyCacheConnection{
									Host: "valkey.svc",
									PasswordSecretRef: &corev1.SecretKeySelector{
										LocalObjectReference: corev1.LocalObjectReference{
											Name: "valkey-credentials",
										},
										Key: "password",
									},
								},
							},
						},
					},
				},
			},
			secrets:     []runtime.Object{valkeySecret},
			expectError: false,
			checkFunc: func(t *testing.T, sr *vllmv1alpha1.SemanticRouter) {
				if sr.Spec.Config.SemanticCache.Valkey.Connection.Password != "valkey-password" {
					t.Errorf("Valkey password not resolved, got %q", sr.Spec.Config.SemanticCache.Valkey.Connection.Password)
				}
				if sr.Spec.Config.SemanticCache.Valkey.Connection.PasswordSecretRef != nil {
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
		{
			name: "error if valkey secret not found",
			sr: &vllmv1alpha1.SemanticRouter{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-router",
					Namespace: "default",
				},
				Spec: vllmv1alpha1.SemanticRouterSpec{
					Config: vllmv1alpha1.ConfigSpec{
						SemanticCache: &vllmv1alpha1.SemanticCacheConfig{
							Enabled:     true,
							BackendType: "valkey",
							Valkey: &vllmv1alpha1.ValkeyCacheConfig{
								Connection: vllmv1alpha1.ValkeyCacheConnection{
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
