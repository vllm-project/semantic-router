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
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

type bootstrapDeploymentContract struct {
	Revision             string
	ManagementStore      bool
	RuntimeStore         bool
	ManagementAPIEnabled bool
	AccessEnabled        bool
	ManagementPort       int32
	BackendDispatchPort  int32
	PostgresDSNEnv       string
	PostgresDSNFile      string
}

func (contract bootstrapDeploymentContract) usesDurableState() bool {
	return contract.ManagementStore
}

func (contract bootstrapDeploymentContract) exposesManagementAPI() bool {
	return contract.ManagementAPIEnabled
}

func (contract bootstrapDeploymentContract) usesBackendDispatch() bool {
	return contract.ManagementStore
}

func (contract bootstrapDeploymentContract) enablesAvailabilityDefaults() bool {
	return contract.ManagementStore
}

// validateBootstrapConfigMap verifies the deployment boundary without
// compiling Router configuration. Provider Integration composition and full
// manifest validation remain application-owned Router startup concerns.
func (r *SemanticRouterReconciler) validateBootstrapConfigMap(
	ctx context.Context,
	sr *vllmv1alpha1.SemanticRouter,
) (bootstrapDeploymentContract, error) {
	ref := sr.Spec.Bootstrap.ConfigMapRef
	if ref.Name == "" || ref.Key == "" {
		return bootstrapDeploymentContract{}, fmt.Errorf("spec.bootstrap.configMapRef.name and key are required")
	}

	configMap := &corev1.ConfigMap{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: sr.Namespace, Name: ref.Name}, configMap); err != nil {
		return bootstrapDeploymentContract{}, fmt.Errorf("get bootstrap ConfigMap %s/%s: %w", sr.Namespace, ref.Name, err)
	}
	if configMap.Immutable == nil || !*configMap.Immutable {
		return bootstrapDeploymentContract{}, fmt.Errorf("bootstrap ConfigMap %s/%s must set immutable: true", sr.Namespace, ref.Name)
	}
	manifest, ok := configMap.Data[ref.Key]
	if !ok {
		return bootstrapDeploymentContract{}, fmt.Errorf("bootstrap ConfigMap %s/%s does not contain data key %q", sr.Namespace, ref.Name, ref.Key)
	}
	contract, err := validateBootstrapManifest([]byte(manifest))
	if err != nil {
		return bootstrapDeploymentContract{}, fmt.Errorf("invalid bootstrap manifest in ConfigMap %s/%s key %q: %w", sr.Namespace, ref.Name, ref.Key, err)
	}
	return contract, nil
}

func validateBootstrapManifest(data []byte) (bootstrapDeploymentContract, error) {
	root, err := decodeBootstrapManifestRoot(data)
	if err != nil {
		return bootstrapDeploymentContract{}, err
	}

	global := mappingNode(root, "global")
	digest := sha256.Sum256(data)
	contract := bootstrapDeploymentContract{
		Revision:            fmt.Sprintf("sha256:%x", digest),
		ManagementPort:      DefaultAPIPort,
		BackendDispatchPort: DefaultBackendDispatchPort,
	}
	stores := mappingNode(global, "stores")
	managementStore := mappingNode(stores, "management")
	runtimeStore := mappingNode(stores, "runtime")
	contract.ManagementStore = managementStore != nil
	contract.RuntimeStore = runtimeStore != nil
	if contract.ManagementStore {
		postgres := mappingNode(managementStore, "postgres")
		if postgres == nil {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.stores.management requires postgres")
		}
		contract.PostgresDSNEnv = strings.TrimSpace(mappingScalar(postgres, "dsn_env"))
		contract.PostgresDSNFile = strings.TrimSpace(mappingScalar(postgres, "dsn_file"))
		if (contract.PostgresDSNEnv == "") == (contract.PostgresDSNFile == "") {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.stores.management.postgres requires exactly one dsn_env or dsn_file")
		}
		if contract.PostgresDSNFile != "" && !strings.HasPrefix(contract.PostgresDSNFile, "/") {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.stores.management.postgres.dsn_file must be an absolute path")
		}
	}
	if contract.RuntimeStore {
		redis := mappingNode(runtimeStore, "redis")
		if redis == nil {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.stores.runtime requires redis")
		}
		urlEnv := strings.TrimSpace(mappingScalar(redis, "url_env"))
		urlFile := strings.TrimSpace(mappingScalar(redis, "url_file"))
		if (urlEnv == "") == (urlFile == "") {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.stores.runtime.redis requires exactly one url_env or url_file")
		}
		if urlFile != "" && !strings.HasPrefix(urlFile, "/") {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.stores.runtime.redis.url_file must be an absolute path")
		}
		if !contract.ManagementStore {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.stores.runtime requires global.stores.management")
		}
	}

	services := mappingNode(global, "services")
	managementAPI := mappingNode(services, "management_api")
	managementEnabled, err := mappingBool(managementAPI, "enabled", false)
	if err != nil {
		return bootstrapDeploymentContract{}, fmt.Errorf("global.services.management_api.enabled: %w", err)
	}
	contract.ManagementAPIEnabled = managementEnabled
	if contract.ManagementAPIEnabled {
		if !contract.ManagementStore {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.services.management_api.enabled requires global.stores.management")
		}
	}
	// The same HTTP listener serves operational probes even when versioned
	// Management routes are disabled. Its port therefore comes from the
	// listener configuration whenever durable routing owns that surface; TLS
	// remains gated by ManagementAPIEnabled.
	if contract.ManagementStore {
		managementPort, err := mappingPort(managementAPI, "port", DefaultAPIPort)
		if err != nil {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.services.management_api.port: %w", err)
		}
		contract.ManagementPort = managementPort
	}
	access := mappingNode(services, "access")
	accessEnabled, err := mappingBool(access, "enabled", false)
	if err != nil {
		return bootstrapDeploymentContract{}, fmt.Errorf("global.services.access.enabled: %w", err)
	}
	contract.AccessEnabled = accessEnabled
	if contract.AccessEnabled && (!contract.ManagementStore || !contract.RuntimeStore) {
		return bootstrapDeploymentContract{}, fmt.Errorf("global.services.access.enabled requires management and runtime stores")
	}
	if contract.usesBackendDispatch() {
		backendDispatch := mappingNode(services, "backend_dispatch")
		backendDispatchPort, err := mappingPort(backendDispatch, "port", DefaultBackendDispatchPort)
		if err != nil {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.services.backend_dispatch.port: %w", err)
		}
		contract.BackendDispatchPort = backendDispatchPort
	}
	return contract, nil
}

func decodeBootstrapManifestRoot(data []byte) (*yaml.Node, error) {
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	var document yaml.Node
	if err := decoder.Decode(&document); err != nil {
		return nil, fmt.Errorf("decode YAML: %w", err)
	}
	var trailing yaml.Node
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err != nil {
			return nil, fmt.Errorf("decode trailing YAML: %w", err)
		}
		return nil, fmt.Errorf("manifest must contain exactly one YAML document")
	}
	root := documentRootMapping(&document)
	if root == nil {
		return nil, fmt.Errorf("manifest must be a YAML mapping")
	}
	if err := validateUniqueMappingKeys(root, "manifest"); err != nil {
		return nil, err
	}
	version := mappingScalar(root, "version")
	if version != "v0.3" {
		return nil, fmt.Errorf("version must be v0.3")
	}
	return root, nil
}

func mappingBool(mapping *yaml.Node, key string, fallback bool) (bool, error) {
	value := strings.TrimSpace(mappingScalar(mapping, key))
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return false, fmt.Errorf("must be true or false")
	}
	return parsed, nil
}

func mappingPort(mapping *yaml.Node, key string, fallback int32) (int32, error) {
	value := strings.TrimSpace(mappingScalar(mapping, key))
	if value == "" {
		return fallback, nil
	}
	port, err := strconv.ParseInt(value, 10, 32)
	if err != nil || port < 1 || port > 65535 {
		return 0, fmt.Errorf("must be an integer from 1 through 65535")
	}
	return int32(port), nil
}

func documentRootMapping(document *yaml.Node) *yaml.Node {
	if document == nil || len(document.Content) != 1 {
		return nil
	}
	root := document.Content[0]
	if root.Kind != yaml.MappingNode {
		return nil
	}
	return root
}

func mappingNode(mapping *yaml.Node, key string) *yaml.Node {
	if mapping == nil || mapping.Kind != yaml.MappingNode {
		return nil
	}
	for index := 0; index+1 < len(mapping.Content); index += 2 {
		if mapping.Content[index].Value == key {
			return mapping.Content[index+1]
		}
	}
	return nil
}

func mappingScalar(mapping *yaml.Node, key string) string {
	node := mappingNode(mapping, key)
	if node == nil || node.Kind != yaml.ScalarNode {
		return ""
	}
	return node.Value
}

func validateUniqueMappingKeys(node *yaml.Node, path string) error {
	if node == nil {
		return nil
	}
	if node.Kind == yaml.MappingNode {
		seen := make(map[string]struct{}, len(node.Content)/2)
		for index := 0; index+1 < len(node.Content); index += 2 {
			key := node.Content[index].Value
			if _, exists := seen[key]; exists {
				return fmt.Errorf("%s contains duplicate key %q", path, key)
			}
			seen[key] = struct{}{}
			if err := validateUniqueMappingKeys(node.Content[index+1], path+"."+key); err != nil {
				return err
			}
		}
		return nil
	}
	for _, child := range node.Content {
		if err := validateUniqueMappingKeys(child, path); err != nil {
			return err
		}
	}
	return nil
}
