package modelruntime

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// Where the Helm chart runs the Router.
const (
	RouterNamespace     = "vllm-semantic-router-system"
	RouterContainer     = "semantic-router"
	routerLabelSelector = "app.kubernetes.io/name=semantic-router,app.kubernetes.io/component=router"
)

// RouterPod returns the running Router pod as an exec target.
func RouterPod(ctx context.Context, client kubernetes.Interface, restConfig *rest.Config) (PodTarget, error) {
	pods, err := client.CoreV1().Pods(RouterNamespace).List(ctx, metav1.ListOptions{LabelSelector: routerLabelSelector})
	if err != nil {
		return PodTarget{}, fmt.Errorf("list router pods: %w", err)
	}
	for _, pod := range pods.Items {
		if pod.Status.Phase == corev1.PodRunning && pod.DeletionTimestamp == nil {
			return PodTarget{Client: client, RestConfig: restConfig, Namespace: RouterNamespace, Pod: pod.Name, Container: RouterContainer}, nil
		}
	}
	return PodTarget{}, fmt.Errorf("no running router pod in %s", RouterNamespace)
}

// ApplyConfigMap creates or replaces a ConfigMap, creating its namespace first
// when needed (profiles prepare files before the Helm release creates it).
func ApplyConfigMap(ctx context.Context, client kubernetes.Interface, namespace, name string, data map[string]string) error {
	if _, err := client.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{}); apierrors.IsNotFound(err) {
		_, err = client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}, metav1.CreateOptions{})
		if err != nil && !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("create namespace %s: %w", namespace, err)
		}
	} else if err != nil {
		return fmt.Errorf("get namespace %s: %w", namespace, err)
	}
	configMap := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace}, Data: data}
	configMaps := client.CoreV1().ConfigMaps(namespace)
	if _, err := configMaps.Create(ctx, configMap, metav1.CreateOptions{}); apierrors.IsAlreadyExists(err) {
		_, err = configMaps.Update(ctx, configMap, metav1.UpdateOptions{})
		return err
	} else if err != nil {
		return fmt.Errorf("create configmap %s/%s: %w", namespace, name, err)
	}
	return nil
}

// DeleteConfigMap removes a ConfigMap; a missing one is not an error.
func DeleteConfigMap(ctx context.Context, client kubernetes.Interface, namespace, name string) error {
	err := client.CoreV1().ConfigMaps(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}
