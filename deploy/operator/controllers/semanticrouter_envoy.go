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
	"reflect"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

const standaloneEnvoyConfigYAML = `static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 8801
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          access_log:
          - name: envoy.access_loggers.stdout
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
          route_config:
            name: local_route
            virtual_hosts:
            - name: local_service
              domains: ["*"]
              routes:
              # Route /v1/models to semantic router HTTP API
              - match:
                  path: "/v1/models"
                route:
                  cluster: semantic_router_cluster
                  timeout: 300s
              {{- range .Models }}
              - match:
                  prefix: "/"
                  headers:
                  - name: x-selected-model
                    string_match:
                      exact: {{ quote .Name }}
                route:
                  weighted_clusters:
                    clusters:
                    {{- range .Backends }}
                    - name: {{ .Name }}
                      weight: {{ .Weight }}
                    {{- end }}
                  auto_host_rewrite: true
                  timeout: 300s
              {{- end }}
              - match:
                  prefix: "/"
                {{- if .DefaultBackends }}
                route:
                  weighted_clusters:
                    clusters:
                    {{- range .DefaultBackends }}
                    - name: {{ .Name }}
                      weight: {{ .Weight }}
                    {{- end }}
                  auto_host_rewrite: true
                  timeout: 300s
                {{- else }}
                direct_response:
                  status: 503
                  body:
                    inline_string: "No model backends are configured"
                {{- end }}
          http_filters:
          # ExtProc filter - semantic router processes requests first
          - name: envoy.filters.http.ext_proc
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
              grpc_service:
                envoy_grpc:
                  cluster_name: extproc_service
              allow_mode_override: true
              processing_mode:
                request_header_mode: "SEND"
                response_header_mode: "SEND"
                request_body_mode: "BUFFERED"
                response_body_mode: "BUFFERED"
                request_trailer_mode: "SKIP"
                response_trailer_mode: "SKIP"
              failure_mode_allow: false
              message_timeout: 300s

          # Router filter (must be last)
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router

          http2_protocol_options:
            max_concurrent_streams: 100
            initial_stream_window_size: 65536
            initial_connection_window_size: 1048576
          stream_idle_timeout: "300s"
          request_timeout: "300s"
          common_http_protocol_options:
            idle_timeout: "300s"

  clusters:
  # ExtProc service - semantic router on localhost:50051
  - name: extproc_service
    connect_timeout: 300s
    per_connection_buffer_limit_bytes: 52428800
    type: STATIC
    lb_policy: ROUND_ROBIN
    typed_extension_protocol_options:
      envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
        "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
        explicit_http_config:
          http2_protocol_options:
            connection_keepalive:
              interval: 300s
              timeout: 300s
    load_assignment:
      cluster_name: extproc_service
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 127.0.0.1
                port_value: 50051

  # Semantic router HTTP API cluster
  - name: semantic_router_cluster
    connect_timeout: 300s
    per_connection_buffer_limit_bytes: 52428800
    type: STATIC
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: semantic_router_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 127.0.0.1
                port_value: 8080
    typed_extension_protocol_options:
      envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
        "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
        explicit_http_config:
          http_protocol_options: {}

  # Each discovered backend has its own cluster; the model route owns weights.
  {{- range .Models }}
  {{- range .Backends }}
  - name: {{ .Name }}
    connect_timeout: 10s
    type: STRICT_DNS
    dns_lookup_family: V4_ONLY
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: {{ .Name }}
      endpoints:
      - lb_endpoints:
        - endpoint:
            hostname: {{ quote .Authority }}
            address:
              socket_address:
                address: {{ quote .Host }}
                port_value: {{ .Port }}
    {{- if .TLS }}
    transport_socket:
      name: envoy.transport_sockets.tls
      typed_config:
        "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext
        sni: {{ quote .Host }}
        common_tls_context:
          validation_context:
            trusted_ca:
              filename: /etc/ssl/certs/ca-certificates.crt
            match_typed_subject_alt_names:
            - san_type: DNS
              matcher:
                exact: {{ quote .Host }}
    {{- end }}
  {{- end }}
  {{- end }}

admin:
  address:
    socket_address:
      address: "127.0.0.1"
      port_value: 19000
`

func (r *SemanticRouterReconciler) reconcileEnvoyConfig(ctx context.Context, sr *vllmv1alpha1.SemanticRouter, gatewayMode string) error {
	if gatewayMode != "standalone" {
		return nil
	}

	canonical, err := r.buildCanonicalConfig(ctx, sr)
	if err != nil {
		return err
	}
	envoyConfig, err := generateStandaloneEnvoyConfig(canonical)
	if err != nil {
		return err
	}

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name + "-envoy-config",
			Namespace: sr.Namespace,
		},
		Data: map[string]string{
			"envoy.yaml": envoyConfig,
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
			if err = r.Get(ctx, types.NamespacedName{Name: cm.Name, Namespace: cm.Namespace}, found); err != nil {
				return err
			}
			found.Data = cm.Data
			return r.Update(ctx, found)
		})
	}

	return nil
}
