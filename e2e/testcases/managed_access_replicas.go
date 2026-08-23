package testcases

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
)

const (
	managedAccessReplicaProxyPort           = 10000
	managedAccessReplicaBackendDispatchPort = 8180
	managedAccessReplicaProxyImage          = "envoyproxy/envoy:v1.35.3"
)

type managedAccessReplicaSession struct {
	baseURL string
	client  *http.Client
	stop    func()
}

func openManagedAccessReplicaSessions(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) ([]managedAccessReplicaSession, func(), error) {
	routerPods, err := managedAccessReadyRouterPods(ctx, client)
	if err != nil {
		return nil, nil, err
	}
	seed, err := managedAccessUUID()
	if err != nil {
		return nil, nil, fmt.Errorf("create replica proxy identity: %w", err)
	}
	seed = strings.ReplaceAll(seed, "-", "")[:12]
	configName := "managed-access-direct-" + seed
	configs := make(map[string]string, len(routerPods))
	for index := range routerPods {
		config, configErr := managedAccessReplicaEnvoyConfig(routerPods[index].Status.PodIP)
		if configErr != nil {
			return nil, nil, configErr
		}
		configs["envoy-"+strconv.Itoa(index)+".yaml"] = config
	}
	configMaps := client.CoreV1().ConfigMaps(managedAccessNamespace)
	if _, err := configMaps.Create(ctx, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName, Namespace: managedAccessNamespace,
			Labels: map[string]string{"vllm.ai/e2e-managed": "true"},
		},
		Data: configs,
	}, metav1.CreateOptions{}); err != nil {
		return nil, nil, fmt.Errorf("create exact-replica proxy configuration: %w", err)
	}

	pods := client.CoreV1().Pods(managedAccessNamespace)
	proxyNames := make([]string, 0, len(routerPods))
	cleanupResources := func() {
		for _, name := range proxyNames {
			_ = pods.Delete(context.Background(), name, metav1.DeleteOptions{})
		}
		_ = configMaps.Delete(context.Background(), configName, metav1.DeleteOptions{})
	}
	for index := range routerPods {
		proxyName := configName + "-" + strconv.Itoa(index)
		proxyNames = append(proxyNames, proxyName)
		if _, err := pods.Create(ctx, managedAccessReplicaProxyPod(
			proxyName, configName, "envoy-"+strconv.Itoa(index)+".yaml",
		), metav1.CreateOptions{}); err != nil {
			cleanupResources()
			return nil, nil, fmt.Errorf("create exact-replica proxy %d: %w", index, err)
		}
	}
	for _, name := range proxyNames {
		if err := waitManagedAccessProxyReady(ctx, client, name); err != nil {
			cleanupResources()
			return nil, nil, err
		}
	}

	sessions := make([]managedAccessReplicaSession, 0, len(proxyNames))
	cleanup := func() {
		for index := range sessions {
			if sessions[index].stop != nil {
				sessions[index].stop()
			}
		}
		cleanupResources()
	}
	for _, name := range proxyNames {
		localPort, err := managedAccessAvailablePort()
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		stop, err := helpers.StartPodPortForward(
			ctx, client, opts.RestConfig, managedAccessNamespace, name,
			localPort+":"+strconv.Itoa(managedAccessReplicaProxyPort), opts.Verbose,
		)
		if err != nil {
			cleanup()
			return nil, nil, fmt.Errorf("open exact-replica proxy session: %w", err)
		}
		sessions = append(sessions, managedAccessReplicaSession{
			baseURL: "http://127.0.0.1:" + localPort,
			client:  &http.Client{Timeout: 45 * time.Second},
			stop:    stop,
		})
	}
	return sessions, cleanup, nil
}

func managedAccessReadyRouterPods(
	ctx context.Context,
	client *kubernetes.Clientset,
) ([]corev1.Pod, error) {
	deployment, err := client.AppsV1().Deployments(managedAccessNamespace).Get(
		ctx, "semantic-router", metav1.GetOptions{},
	)
	if err != nil {
		return nil, fmt.Errorf("read Router deployment for exact-replica sessions: %w", err)
	}
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return nil, fmt.Errorf("resolve Router replica selector: %w", err)
	}
	podList, err := client.CoreV1().Pods(managedAccessNamespace).List(
		ctx, metav1.ListOptions{LabelSelector: selector.String()},
	)
	if err != nil {
		return nil, fmt.Errorf("list Router replicas: %w", err)
	}
	ready := make([]corev1.Pod, 0, len(podList.Items))
	for index := range podList.Items {
		pod := podList.Items[index]
		if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" && managedAccessPodReady(pod) {
			ready = append(ready, pod)
		}
	}
	if len(ready) != 2 {
		return nil, fmt.Errorf("found %d ready Router Pods for exact-replica sessions, want 2", len(ready))
	}
	sort.Slice(ready, func(left, right int) bool { return ready[left].Name < ready[right].Name })
	return ready, nil
}

func managedAccessPodReady(pod corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func managedAccessReplicaProxyPod(name string, configName string, configKey string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: managedAccessNamespace,
			Labels: map[string]string{
				"vllm.ai/e2e-managed":  "true",
				"vllm.ai/e2e-contract": "managed-access",
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{{
				Name: "envoy", Image: managedAccessReplicaProxyImage,
				ImagePullPolicy: corev1.PullIfNotPresent,
				Args:            []string{"-c", "/etc/envoy/envoy.yaml", "--log-level", "warning"},
				Ports: []corev1.ContainerPort{{
					Name: "http", ContainerPort: managedAccessReplicaProxyPort,
				}},
				ReadinessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{TCPSocket: &corev1.TCPSocketAction{
						Port: intstr.FromInt(managedAccessReplicaProxyPort),
					}},
					PeriodSeconds: 1, TimeoutSeconds: 1, FailureThreshold: 30,
				},
				VolumeMounts: []corev1.VolumeMount{{
					Name: "config", MountPath: "/etc/envoy/envoy.yaml", SubPath: "envoy.yaml", ReadOnly: true,
				}},
			}},
			Volumes: []corev1.Volume{{
				Name: "config",
				VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
					LocalObjectReference: corev1.LocalObjectReference{Name: configName},
					Items:                []corev1.KeyToPath{{Key: configKey, Path: "envoy.yaml"}},
				}},
			}},
		},
	}
}

func waitManagedAccessProxyReady(
	ctx context.Context,
	client *kubernetes.Clientset,
	name string,
) error {
	return wait.PollUntilContextTimeout(ctx, time.Second, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
		pod, err := client.CoreV1().Pods(managedAccessNamespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if managedAccessPodReady(*pod) {
			return true, nil
		}
		for _, status := range pod.Status.ContainerStatuses {
			if status.State.Terminated != nil {
				return false, fmt.Errorf(
					"exact-replica proxy %s terminated: %s", name, status.State.Terminated.Message,
				)
			}
			if status.State.Waiting != nil && (status.State.Waiting.Reason == "ErrImagePull" ||
				status.State.Waiting.Reason == "ImagePullBackOff" || status.State.Waiting.Reason == "InvalidImageName") {
				return false, fmt.Errorf(
					"exact-replica proxy %s cannot start: %s", name, status.State.Waiting.Reason,
				)
			}
		}
		return false, nil
	})
}

func managedAccessReplicaEnvoyConfig(routerPodIP string) (string, error) {
	if net.ParseIP(routerPodIP) == nil {
		return "", fmt.Errorf("Router replica has invalid Pod IP")
	}
	return fmt.Sprintf(`static_resources:
  listeners:
  - name: managed_access_ingress
    address:
      socket_address: { address: 0.0.0.0, port_value: %d }
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: managed_access
          stream_idle_timeout: 60s
          route_config:
            name: managed_access_route
            virtual_hosts:
            - name: backend
              domains: ["*"]
              routes:
              - match: { prefix: "/" }
                route: { cluster: inference_backend, timeout: 60s }
          http_filters:
          - name: envoy.filters.http.ext_proc
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
              grpc_service:
                envoy_grpc: { cluster_name: exact_router_replica }
              failure_mode_allow: false
              message_timeout: 60s
              processing_mode:
                request_header_mode: SEND
                response_header_mode: SEND
                request_body_mode: BUFFERED
                response_body_mode: BUFFERED
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
  clusters:
  - name: exact_router_replica
    connect_timeout: 5s
    type: STATIC
    http2_protocol_options: {}
    load_assignment:
      cluster_name: exact_router_replica
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address: { address: %s, port_value: 50051 }
  - name: inference_backend
    connect_timeout: 5s
    type: STATIC
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: inference_backend
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address: { address: %s, port_value: %d }
`, managedAccessReplicaProxyPort, routerPodIP, routerPodIP, managedAccessReplicaBackendDispatchPort), nil
}

func waitManagedAccessReplicaDiscovery(
	ctx context.Context,
	sessions []managedAccessReplicaSession,
	secret string,
	authorized string,
	hidden string,
) error {
	if len(sessions) != 2 {
		return fmt.Errorf("exact-replica discovery requires two sessions")
	}
	for index := range sessions {
		if err := waitManagedAccessDiscovery(
			ctx, sessions[index].client, sessions[index].baseURL, secret, authorized, hidden,
		); err != nil {
			return fmt.Errorf("Router replica %d discovery: %w", index, err)
		}
	}
	return nil
}

func waitManagedAccessReplicaCredentialDenied(
	ctx context.Context,
	sessions []managedAccessReplicaSession,
	secret string,
) error {
	if len(sessions) != 2 {
		return fmt.Errorf("exact-replica credential check requires two sessions")
	}
	for index := range sessions {
		if err := waitManagedAccessCredentialDenied(
			ctx, sessions[index].client, sessions[index].baseURL, secret,
		); err != nil {
			return fmt.Errorf("Router replica %d credential denial: %w", index, err)
		}
	}
	return nil
}

func managedAccessReplicaBurst(
	ctx context.Context,
	sessions []managedAccessReplicaSession,
	secret string,
	model string,
	count int,
) []managedAccessInvocation {
	results := make([]managedAccessInvocation, count)
	if len(sessions) != 2 {
		for index := range results {
			results[index].err = fmt.Errorf("managed-access burst requires two exact-replica sessions")
		}
		return results
	}
	for index := range results {
		session := sessions[index%len(sessions)]
		results[index] = invokeManagedAccessModel(
			ctx, session.client, session.baseURL, secret, model, index,
		)
	}
	return results
}
