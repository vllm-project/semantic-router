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

const (
	controlPlaneModeStandalone = "standalone"
	controlPlaneModeManaged    = "managed"
)

type bootstrapDeploymentContract struct {
	Mode                string
	Revision            string
	ManagementPort      int32
	BackendDispatchPort int32
	PostgresDSNEnv      string
	PostgresDSNFile     string
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
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	var document yaml.Node
	if err := decoder.Decode(&document); err != nil {
		return bootstrapDeploymentContract{}, fmt.Errorf("decode YAML: %w", err)
	}
	var trailing yaml.Node
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err != nil {
			return bootstrapDeploymentContract{}, fmt.Errorf("decode trailing YAML: %w", err)
		}
		return bootstrapDeploymentContract{}, fmt.Errorf("manifest must contain exactly one YAML document")
	}
	root := documentRootMapping(&document)
	if root == nil {
		return bootstrapDeploymentContract{}, fmt.Errorf("manifest must be a YAML mapping")
	}
	if err := validateUniqueMappingKeys(root, "manifest"); err != nil {
		return bootstrapDeploymentContract{}, err
	}
	version := mappingScalar(root, "version")
	if version != "v0.4" {
		return bootstrapDeploymentContract{}, fmt.Errorf("version must be v0.4")
	}

	global := mappingNode(root, "global")
	controlPlane := mappingNode(global, "control_plane")
	mode := strings.TrimSpace(mappingScalar(controlPlane, "mode"))
	if mode != controlPlaneModeStandalone && mode != controlPlaneModeManaged {
		return bootstrapDeploymentContract{}, fmt.Errorf("global.control_plane.mode must be standalone or managed")
	}
	digest := sha256.Sum256(data)
	contract := bootstrapDeploymentContract{
		Mode:                mode,
		Revision:            fmt.Sprintf("sha256:%x", digest),
		ManagementPort:      DefaultAPIPort,
		BackendDispatchPort: DefaultBackendDispatchPort,
	}
	if mode == controlPlaneModeManaged {
		for _, field := range []string{"models", "recipes", "entrypoints"} {
			if mappingNode(root, field) != nil {
				return bootstrapDeploymentContract{}, fmt.Errorf("managed bootstrap must not declare top-level %s", field)
			}
		}

		stores := mappingNode(global, "stores")
		accessStore := mappingNode(stores, "access")
		postgres := mappingNode(accessStore, "postgres")
		contract.PostgresDSNEnv = strings.TrimSpace(mappingScalar(postgres, "dsn_env"))
		contract.PostgresDSNFile = strings.TrimSpace(mappingScalar(postgres, "dsn_file"))
		if (contract.PostgresDSNEnv == "") == (contract.PostgresDSNFile == "") {
			return bootstrapDeploymentContract{}, fmt.Errorf("managed bootstrap requires exactly one global.stores.access.postgres dsn_env or dsn_file")
		}
		if contract.PostgresDSNFile != "" && !strings.HasPrefix(contract.PostgresDSNFile, "/") {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.stores.access.postgres.dsn_file must be an absolute path")
		}

		services := mappingNode(global, "services")
		managementAPI := mappingNode(services, "management_api")
		managementPort, err := mappingPort(managementAPI, "port", DefaultAPIPort)
		if err != nil {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.services.management_api.port: %w", err)
		}
		contract.ManagementPort = managementPort
		backendDispatch := mappingNode(services, "backend_dispatch")
		backendDispatchPort, err := mappingPort(backendDispatch, "port", DefaultBackendDispatchPort)
		if err != nil {
			return bootstrapDeploymentContract{}, fmt.Errorf("global.services.backend_dispatch.port: %w", err)
		}
		contract.BackendDispatchPort = backendDispatchPort
	}
	return contract, nil
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
