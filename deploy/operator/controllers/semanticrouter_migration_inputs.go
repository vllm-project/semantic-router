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
	"fmt"
	"path/filepath"
	"strings"

	corev1 "k8s.io/api/core/v1"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

// migrationPodInputs limits the one-shot migrator to the PostgreSQL DSN
// projection it consumes. Router credentials, Valkey credentials, and runtime
// configuration never enter the migration Pod.
type migrationPodInputs struct {
	environment  []corev1.EnvVar
	volumes      []corev1.Volume
	volumeMounts []corev1.VolumeMount
}

func resolveMigrationPodInputs(
	sr *vllmv1alpha1.SemanticRouter,
	bootstrap bootstrapDeploymentContract,
) (migrationPodInputs, error) {
	if bootstrap.PostgresDSNEnv != "" {
		environment, err := migrationDSNEnvironment(sr, bootstrap.PostgresDSNEnv)
		return migrationPodInputs{environment: environment}, err
	}
	if bootstrap.PostgresDSNFile == "" {
		return migrationPodInputs{}, fmt.Errorf("Management migration requires a PostgreSQL DSN source")
	}
	volume, mount, err := migrationDSNFileProjection(sr, bootstrap.PostgresDSNFile)
	if err != nil {
		return migrationPodInputs{}, err
	}
	return migrationPodInputs{
		volumes:      []corev1.Volume{volume},
		volumeMounts: []corev1.VolumeMount{mount},
	}, nil
}

func migrationDSNEnvironment(
	sr *vllmv1alpha1.SemanticRouter,
	name string,
) ([]corev1.EnvVar, error) {
	matches := make([]corev1.EnvVar, 0, 1)
	for _, variable := range sr.Spec.Env {
		if variable.Name == name {
			matches = append(matches, variable)
		}
	}
	if len(matches) == 1 {
		if matches[0].ValueFrom == nil || matches[0].ValueFrom.SecretKeyRef == nil {
			return nil, fmt.Errorf("PostgreSQL DSN environment variable %s must use a Secret key reference", name)
		}
		return matches, nil
	}
	if len(matches) > 1 {
		return nil, fmt.Errorf("PostgreSQL DSN environment variable %s is declared more than once", name)
	}

	// A single unprefixed Secret envFrom source has deterministic Kubernetes
	// key-to-variable semantics. Project only the DSN key instead of inheriting
	// every key from that Secret into the migrator.
	secretSources := make([]*corev1.SecretEnvSource, 0, 1)
	for _, source := range sr.Spec.EnvFrom {
		if source.Prefix == "" && source.SecretRef != nil {
			secretSources = append(secretSources, source.SecretRef)
		}
	}
	if len(secretSources) != 1 {
		return nil, fmt.Errorf(
			"PostgreSQL DSN environment variable %s requires one explicit env Secret reference or one unprefixed envFrom Secret",
			name,
		)
	}
	secret := secretSources[0]
	return []corev1.EnvVar{{
		Name: name,
		ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{
			LocalObjectReference: secret.LocalObjectReference,
			Key:                  name,
			Optional:             secret.Optional,
		}},
	}}, nil
}

func migrationDSNFileProjection(
	sr *vllmv1alpha1.SemanticRouter,
	dsnFile string,
) (corev1.Volume, corev1.VolumeMount, error) {
	cleanFile := filepath.Clean(dsnFile)
	bestIndex := -1
	bestLength := -1
	for index, mount := range sr.Spec.VolumeMounts {
		if !mount.ReadOnly || !volumeMountContainsPath(mount, cleanFile) {
			continue
		}
		length := len(filepath.Clean(mount.MountPath))
		if length > bestLength {
			bestIndex = index
			bestLength = length
		}
	}
	if bestIndex < 0 {
		return corev1.Volume{}, corev1.VolumeMount{}, fmt.Errorf(
			"PostgreSQL DSN file %s requires a matching read-only volume mount",
			dsnFile,
		)
	}
	mount := sr.Spec.VolumeMounts[bestIndex]
	for _, volume := range sr.Spec.Volumes {
		if volume.Name == mount.Name {
			if !migrationSecretVolume(volume) {
				return corev1.Volume{}, corev1.VolumeMount{}, fmt.Errorf(
					"PostgreSQL DSN file mount %s must use a Secret, Secret-only projected, or CSI volume",
					mount.Name,
				)
			}
			return volume, mount, nil
		}
	}
	return corev1.Volume{}, corev1.VolumeMount{}, fmt.Errorf(
		"PostgreSQL DSN file mount %s references an undeclared volume",
		mount.Name,
	)
}

func volumeMountContainsPath(mount corev1.VolumeMount, target string) bool {
	mountPath := filepath.Clean(mount.MountPath)
	if mount.SubPath != "" || mount.SubPathExpr != "" {
		return mountPath == target
	}
	if mountPath == string(filepath.Separator) {
		return strings.HasPrefix(target, mountPath)
	}
	return target == mountPath || strings.HasPrefix(target, mountPath+string(filepath.Separator))
}

func migrationSecretVolume(volume corev1.Volume) bool {
	if volume.Secret != nil || volume.CSI != nil {
		return true
	}
	if volume.Projected == nil || len(volume.Projected.Sources) == 0 {
		return false
	}
	for _, source := range volume.Projected.Sources {
		if source.Secret == nil {
			return false
		}
	}
	return true
}
