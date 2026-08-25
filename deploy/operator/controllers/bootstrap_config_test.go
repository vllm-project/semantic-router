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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func TestValidateBootstrapManifestCapabilityBoundary(t *testing.T) {
	tests := []struct {
		name    string
		yaml    string
		wantErr string
	}{
		{
			name: "file routing needs no stores",
			yaml: "version: v0.3\nproviders: {}\nrouting: {}\nrecipes: []\nentrypoints: []\n",
		},
		{
			name: "durable routing may include a seed",
			yaml: "version: v0.3\nproviders: {}\nrouting: {}\nrecipes: []\nentrypoints: []\nglobal:\n  stores:\n    management:\n      postgres:\n        dsn_env: TEST_DATABASE_URL\n  services:\n    management_api:\n      enabled: false\n      port: 9080\n",
		},
		{
			name:    "management store requires one DSN source",
			yaml:    "version: v0.3\nglobal:\n  stores:\n    management:\n      postgres: {}\n",
			wantErr: "requires exactly one dsn_env or dsn_file",
		},
		{
			name:    "runtime store requires Management store",
			yaml:    "version: v0.3\nglobal:\n  stores:\n    runtime:\n      redis:\n        url_env: TEST_REDIS_URL\n",
			wantErr: "runtime requires global.stores.management",
		},
		{
			name:    "Management API requires Management store",
			yaml:    "version: v0.3\nglobal:\n  services:\n    management_api:\n      enabled: true\n",
			wantErr: "management_api.enabled requires global.stores.management",
		},
		{
			name:    "access requires both stores",
			yaml:    "version: v0.3\nglobal:\n  stores:\n    management:\n      postgres:\n        dsn_env: TEST_DATABASE_URL\n  services:\n    access:\n      enabled: true\n",
			wantErr: "access.enabled requires management and runtime stores",
		},
		{
			name:    "requires v0.3",
			yaml:    "version: v0.4\n",
			wantErr: "version must be v0.3",
		},
		{
			name:    "rejects multiple documents",
			yaml:    "version: v0.3\n---\nversion: v0.3\n",
			wantErr: "exactly one YAML document",
		},
		{
			name:    "rejects ambiguous duplicate keys",
			yaml:    "version: v0.3\nglobal:\n  stores: {}\n  stores: {}\n",
			wantErr: "duplicate key \"stores\"",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			contract, err := validateBootstrapManifest([]byte(test.yaml))
			if test.wantErr == "" && err != nil {
				t.Fatalf("validateBootstrapManifest() error = %v", err)
			}
			if test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)) {
				t.Fatalf("validateBootstrapManifest() error = %v, want substring %q", err, test.wantErr)
			}
			if test.name == "durable routing may include a seed" &&
				(contract.ManagementPort != 9080 || contract.ManagementAPIEnabled) {
				t.Fatalf("store-only listener contract = %+v", contract)
			}
		})
	}
}

func TestValidateBootstrapConfigMapRequiresImmutableSelectedKey(t *testing.T) {
	routerScheme := runtime.NewScheme()
	if err := scheme.AddToScheme(routerScheme); err != nil {
		t.Fatal(err)
	}
	if err := vllmv1alpha1.AddToScheme(routerScheme); err != nil {
		t.Fatal(err)
	}
	immutable := true
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: "router-bootstrap-v1", Namespace: "default"},
		Immutable:  &immutable,
		Data: map[string]string{
			"router.yaml": "version: v0.3\nglobal:\n  stores:\n    management:\n      postgres:\n        dsn_env: TEST_DATABASE_URL\n",
		},
	}
	router := &vllmv1alpha1.SemanticRouter{
		ObjectMeta: metav1.ObjectMeta{Name: "router", Namespace: "default"},
		Spec: vllmv1alpha1.SemanticRouterSpec{Bootstrap: vllmv1alpha1.BootstrapSpec{
			ConfigMapRef: vllmv1alpha1.BootstrapConfigMapReference{Name: configMap.Name, Key: "router.yaml"},
		}},
	}
	reconciler := &SemanticRouterReconciler{
		Client: fake.NewClientBuilder().WithScheme(routerScheme).WithObjects(configMap).Build(),
		Scheme: routerScheme,
	}
	if _, err := reconciler.validateBootstrapConfigMap(context.Background(), router); err != nil {
		t.Fatalf("validateBootstrapConfigMap() error = %v", err)
	}

	mutable := configMap.DeepCopy()
	mutable.Name = "router-bootstrap-mutable"
	mutable.Immutable = nil
	router.Spec.Bootstrap.ConfigMapRef.Name = mutable.Name
	reconciler.Client = fake.NewClientBuilder().WithScheme(routerScheme).WithObjects(mutable).Build()
	if _, err := reconciler.validateBootstrapConfigMap(context.Background(), router); err == nil || !strings.Contains(err.Error(), "immutable: true") {
		t.Fatalf("mutable ConfigMap error = %v", err)
	}
}

func TestGenerateVolumesMountsOnlySelectedBootstrapKey(t *testing.T) {
	reconciler := &SemanticRouterReconciler{}
	router := &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{
		Bootstrap: vllmv1alpha1.BootstrapSpec{ConfigMapRef: vllmv1alpha1.BootstrapConfigMapReference{
			Name: "router-bootstrap-v7",
			Key:  "production.yaml",
		}},
	}}
	volumes := reconciler.generateVolumes(router, gatewayModeExternal)
	requireSelectedBootstrapConfig(t, volumes[0].ConfigMap, "router-bootstrap-v7", "production.yaml")

	mount := reconciler.generateVolumeMounts(router)[0]
	requireBootstrapVolumeMount(t, mount)

	container := reconciler.buildSemanticRouterContainer(router, bootstrapDeploymentContract{
		ManagementStore: true, ManagementAPIEnabled: true, ManagementPort: 9443, BackendDispatchPort: 8181,
	})
	ownedEnvironment := map[string]string{}
	for _, variable := range container.Env {
		ownedEnvironment[variable.Name] = variable.Value
	}
	requireOperatorEnvironment(t, ownedEnvironment)
}

func requireSelectedBootstrapConfig(
	t *testing.T,
	configVolume *corev1.ConfigMapVolumeSource,
	wantName string,
	wantKey string,
) {
	t.Helper()
	if configVolume == nil {
		t.Fatal("config volume is nil")
	}
	if configVolume.Name != wantName {
		t.Fatalf("config volume name = %q, want %q", configVolume.Name, wantName)
	}
	if len(configVolume.Items) != 1 {
		t.Fatalf("config volume items = %#v, want one selected key", configVolume.Items)
	}
	item := configVolume.Items[0]
	if item.Key != wantKey {
		t.Fatalf("config volume key = %q, want %q", item.Key, wantKey)
	}
	if item.Path != "config.yaml" {
		t.Fatalf("config volume path = %q, want config.yaml", item.Path)
	}
}

func requireBootstrapVolumeMount(t *testing.T, mount corev1.VolumeMount) {
	t.Helper()
	if mount.Name != "config-volume" {
		t.Fatalf("bootstrap volume mount name = %q", mount.Name)
	}
	if mount.MountPath != "/app/config.yaml" {
		t.Fatalf("bootstrap volume mount path = %q", mount.MountPath)
	}
	if mount.SubPath != "config.yaml" {
		t.Fatalf("bootstrap volume subpath = %q", mount.SubPath)
	}
	if !mount.ReadOnly {
		t.Fatal("bootstrap volume mount must be read-only")
	}
}

func requireOperatorEnvironment(t *testing.T, environment map[string]string) {
	t.Helper()
	if environment["CONFIG_FILE"] != "/app/config.yaml" {
		t.Fatalf("CONFIG_FILE = %q", environment["CONFIG_FILE"])
	}
	if environment[managementInternalListenerEnv] != "true" {
		t.Fatalf("%s = %q", managementInternalListenerEnv, environment[managementInternalListenerEnv])
	}
}
