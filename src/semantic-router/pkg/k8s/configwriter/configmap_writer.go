package configwriter

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
)

// Environment variables a Kubernetes-managed deployment sets to point a
// caller at the ConfigMap that backs its mounted config file. Their absence
// means the deployment is not Kubernetes-managed (local, VM, or Docker), so
// callers fall back to writing the local file exactly as before.
const (
	ConfigMapNameEnv      = "VLLM_SR_K8S_CONFIGMAP_NAME"
	ConfigMapNamespaceEnv = "VLLM_SR_K8S_CONFIGMAP_NAMESPACE"
	ConfigMapKeyEnv       = "VLLM_SR_K8S_CONFIGMAP_KEY"

	defaultConfigMapKey = "config.yaml"
)

// ConfigMapTarget identifies the ConfigMap key a canonical document should be
// written to.
type ConfigMapTarget struct {
	Namespace string
	Name      string
	Key       string
}

// ConfigMapTargetFromEnv resolves a write target from the environment. ok is
// false when the deployment has not declared one, which is the case for every
// non-Kubernetes deployment and for a Kubernetes deployment that has not
// opted in (see deploy/helm/semantic-router/templates for the wiring).
func ConfigMapTargetFromEnv() (ConfigMapTarget, bool) {
	name := strings.TrimSpace(os.Getenv(ConfigMapNameEnv))
	namespace := strings.TrimSpace(os.Getenv(ConfigMapNamespaceEnv))
	if name == "" || namespace == "" {
		return ConfigMapTarget{}, false
	}
	key := strings.TrimSpace(os.Getenv(ConfigMapKeyEnv))
	if key == "" {
		key = defaultConfigMapKey
	}
	return ConfigMapTarget{Namespace: namespace, Name: name, Key: key}, true
}

// ErrConfigMapControllerOwned is returned when the target ConfigMap carries a
// controller owner reference (set by, for example, the Operator's
// SemanticRouter reconciler via controllerutil.SetControllerReference). That
// controller regenerates the ConfigMap from its own source of truth on every
// reconcile, so a direct write here would be silently reverted. The caller
// should refuse the request and point the operator at that source instead
// (the SemanticRouter custom resource) rather than attempt the write.
var ErrConfigMapControllerOwned = errors.New("config ConfigMap is owned by a controller; edit its source instead of writing to it directly")

// ErrConfigMapChanged means another writer changed the config document since
// the caller read it. The caller must rebuild its update against the new
// document rather than silently overwrite that writer's change.
var ErrConfigMapChanged = errors.New("config ConfigMap changed during the update")

// ConfigMapWriter persists a canonical config document into a Kubernetes
// ConfigMap key, for deployments where the mounted config file is read-only
// (every shipped manifest mounts it that way; see issue #3688).
type ConfigMapWriter struct {
	clientset kubernetes.Interface
}

// NewInClusterConfigMapWriter builds a ConfigMapWriter from the pod's own
// service account. It returns an error when not running in a cluster (local
// CLI, VM, or plain Docker runs), which callers should treat the same as
// ConfigMapTargetFromEnv returning ok=false: fall back to the local file.
func NewInClusterConfigMapWriter() (*ConfigMapWriter, error) {
	restConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("not running in a Kubernetes cluster: %w", err)
	}
	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("build Kubernetes client: %w", err)
	}
	return &ConfigMapWriter{clientset: clientset}, nil
}

// NewConfigMapWriter builds a ConfigMapWriter around an existing clientset.
// Tests inject a fake clientset here instead of NewInClusterConfigMapWriter.
func NewConfigMapWriter(clientset kubernetes.Interface) *ConfigMapWriter {
	return &ConfigMapWriter{clientset: clientset}
}

// Write sets target.Key to data in the target ConfigMap. It reads the current
// object first so a controller-owned ConfigMap can be detected and refused,
// and retries the update on a conflicting concurrent write (another replica's
// request, or a controller reconcile that landed between the read and the
// write), matching the retry pattern the Operator's own reconciler uses.
func (w *ConfigMapWriter) Write(ctx context.Context, target ConfigMapTarget, data []byte) error {
	return w.write(ctx, target, data, nil)
}

// WriteIfUnchanged atomically replaces a document only when its current value
// still matches expected, including after a Kubernetes resource-version retry.
func (w *ConfigMapWriter) WriteIfUnchanged(ctx context.Context, target ConfigMapTarget, expected, data []byte) error {
	return w.write(ctx, target, data, &expected)
}

func (w *ConfigMapWriter) write(ctx context.Context, target ConfigMapTarget, data []byte, expected *[]byte) error {
	if w == nil || w.clientset == nil {
		return errors.New("k8s: nil ConfigMapWriter")
	}
	configMaps := w.clientset.CoreV1().ConfigMaps(target.Namespace)

	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		current, err := configMaps.Get(ctx, target.Name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return fmt.Errorf("ConfigMap %s/%s not found: %w", target.Namespace, target.Name, err)
			}
			return fmt.Errorf("get ConfigMap %s/%s: %w", target.Namespace, target.Name, err)
		}
		if isControllerOwned(current) {
			return ErrConfigMapControllerOwned
		}
		if expected != nil {
			value, found := current.Data[target.Key]
			if !found || value != string(*expected) {
				return ErrConfigMapChanged
			}
		}

		updated := current.DeepCopy()
		if updated.Data == nil {
			updated.Data = map[string]string{}
		}
		updated.Data[target.Key] = string(data)

		_, err = configMaps.Update(ctx, updated, metav1.UpdateOptions{})
		return err
	})
}

// Read returns target.Key's current value. found is false when the ConfigMap
// exists but has no such key, which is the state before the first write.
func (w *ConfigMapWriter) Read(ctx context.Context, target ConfigMapTarget) (data []byte, found bool, err error) {
	if w == nil || w.clientset == nil {
		return nil, false, errors.New("k8s: nil ConfigMapWriter")
	}
	current, err := w.clientset.CoreV1().ConfigMaps(target.Namespace).Get(ctx, target.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil, false, fmt.Errorf("ConfigMap %s/%s not found: %w", target.Namespace, target.Name, err)
		}
		return nil, false, fmt.Errorf("get ConfigMap %s/%s: %w", target.Namespace, target.Name, err)
	}
	value, ok := current.Data[target.Key]
	if !ok {
		return nil, false, nil
	}
	return []byte(value), true, nil
}

// isControllerOwned reports whether cm names a controller in its owner
// references, the same field controllerutil.SetControllerReference sets.
func isControllerOwned(cm *corev1.ConfigMap) bool {
	for _, owner := range cm.OwnerReferences {
		if owner.Controller != nil && *owner.Controller {
			return true
		}
	}
	return false
}
