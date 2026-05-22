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
	"encoding/json"
	"fmt"
	"reflect"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	"gopkg.in/yaml.v3"
)

func (r *SemanticRouterReconciler) reconcileConfigMap(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	if err := validateSemanticCacheConfig(sr.Spec.Config.SemanticCache); err != nil {
		return fmt.Errorf("invalid semantic cache configuration: %w", err)
	}

	if err := r.resolveSemanticCacheSecrets(ctx, sr); err != nil {
		return fmt.Errorf("failed to resolve cache secrets: %w", err)
	}

	configData, err := r.generateConfigYAML(ctx, sr)
	if err != nil {
		return err
	}

	toolsData, err := r.generateToolsJSON(sr)
	if err != nil {
		return err
	}

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name + "-config",
			Namespace: sr.Namespace,
		},
		Data: map[string]string{
			"config.yaml":   configData,
			"tools_db.json": toolsData,
		},
	}

	if err := controllerutil.SetControllerReference(sr, cm, r.Scheme); err != nil {
		return err
	}

	found := &corev1.ConfigMap{}
	err = r.Get(ctx, types.NamespacedName{Name: cm.Name, Namespace: cm.Namespace}, found)
	if err != nil && errors.IsNotFound(err) {
		return r.Create(ctx, cm)
	} else if err != nil {
		return err
	}

	if !reflect.DeepEqual(found.Data, cm.Data) {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			if err := r.Get(ctx, types.NamespacedName{Name: cm.Name, Namespace: cm.Namespace}, found); err != nil {
				return err
			}
			found.Data = cm.Data
			return r.Update(ctx, found)
		})
	}

	return nil
}

func (r *SemanticRouterReconciler) generateConfigYAML(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) (string, error) {
	config, err := r.buildCanonicalConfig(ctx, sr)
	if err != nil {
		return "", err
	}

	data, err := yaml.Marshal(config)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

func (r *SemanticRouterReconciler) generateToolsJSON(sr *vllmv1alpha1.SemanticRouter) (string, error) {
	if sr.Spec.ToolsDb == nil {
		return "[]", nil
	}

	data, err := json.Marshal(sr.Spec.ToolsDb)
	if err != nil {
		return "", err
	}

	return string(data), nil
}
