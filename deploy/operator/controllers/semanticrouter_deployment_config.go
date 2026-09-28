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
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"maps"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

const deploymentConfigChecksumAnnotation = "vllm.ai/config-checksum"

// Envoy reads its static bootstrap at startup. Roll both containers together
// when their mounted configuration changes so backend discovery and Router
// model selection continue to use the same configuration revision.
func (r *SemanticRouterReconciler) annotateDeploymentConfig(
	ctx context.Context,
	sr *vllmv1alpha1.SemanticRouter,
	gatewayMode string,
	deployment *appsv1.Deployment,
) error {
	names := []string{sr.Name + "-config"}
	if gatewayMode == "standalone" {
		names = append(names, sr.Name+"-envoy-config")
	}
	hash := sha256.New()
	encoder := json.NewEncoder(hash)
	for _, name := range names {
		cm := &corev1.ConfigMap{}
		if err := r.Get(ctx, types.NamespacedName{Namespace: sr.Namespace, Name: name}, cm); err != nil {
			return fmt.Errorf("read deployment configuration %s: %w", name, err)
		}
		// JSON orders map keys. Ignore object metadata to avoid a rollout on
		// resource-version or annotation updates that leave the files intact.
		if err := encoder.Encode(struct {
			Name       string
			Data       map[string]string
			BinaryData map[string][]byte
		}{name, cm.Data, cm.BinaryData}); err != nil {
			return fmt.Errorf("hash deployment configuration %s: %w", name, err)
		}
	}
	annotations := maps.Clone(deployment.Spec.Template.Annotations)
	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[deploymentConfigChecksumAnnotation] = fmt.Sprintf("%x", hash.Sum(nil))
	deployment.Spec.Template.Annotations = annotations
	return nil
}
