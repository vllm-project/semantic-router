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
	"os"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/yaml"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func probedRouter(args ...string) *vllmv1alpha1.SemanticRouter {
	enabled := true
	probe := &vllmv1alpha1.ProbeSpec{Enabled: &enabled}
	return &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{
		Args:           args,
		StartupProbe:   probe,
		LivenessProbe:  probe,
		ReadinessProbe: probe,
	}}
}

func TestRouterContainerServesItsGatewayMode(t *testing.T) {
	r := &SemanticRouterReconciler{}
	listener := intstr.FromString(DefaultListenerName)
	for _, test := range []struct {
		mode     string
		args     []string
		traffic  corev1.ContainerPort
		ready    corev1.ProbeHandler
		liveness corev1.ProbeHandler
		service  corev1.ServicePort
	}{
		{
			mode:    GatewayModeStandalone,
			args:    []string{"-gateway=standalone", "-listener-address=0.0.0.0", "--secure=false"},
			traffic: corev1.ContainerPort{Name: DefaultListenerName, ContainerPort: 8801, Protocol: corev1.ProtocolTCP},
			ready: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{
				Path: "/ready", Port: listener, Scheme: corev1.URISchemeHTTP,
			}},
			liveness: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{
				Path: "/health", Port: listener, Scheme: corev1.URISchemeHTTP,
			}},
			service: corev1.ServicePort{Name: DefaultListenerName, Port: 8801, TargetPort: listener, Protocol: corev1.ProtocolTCP},
		},
		{
			mode:     GatewayModeIntegration,
			args:     []string{"-gateway=extproc", "--secure=false"},
			traffic:  corev1.ContainerPort{Name: "grpc", ContainerPort: 50051, Protocol: corev1.ProtocolTCP},
			ready:    corev1.ProbeHandler{GRPC: &corev1.GRPCAction{Port: 50051}},
			liveness: corev1.ProbeHandler{TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt(50051)}},
			service: corev1.ServicePort{
				Name: "grpc", Port: 50051, TargetPort: intstr.FromInt32(50051), Protocol: corev1.ProtocolTCP,
			},
		},
	} {
		t.Run(test.mode, func(t *testing.T) {
			containers := r.generateContainers(probedRouter("--secure=false"), test.mode)
			if len(containers) != 1 {
				t.Fatalf("the Pod runs the Router alone, got %d containers", len(containers))
			}
			container := containers[0]
			if !reflect.DeepEqual(container.Args, test.args) {
				t.Fatalf("args = %v, want %v (mode flags first, spec.args after)", container.Args, test.args)
			}
			if container.Ports[0] != test.traffic {
				t.Fatalf("traffic port = %+v, want %+v", container.Ports[0], test.traffic)
			}
			if !reflect.DeepEqual(container.StartupProbe.ProbeHandler, test.ready) ||
				!reflect.DeepEqual(container.ReadinessProbe.ProbeHandler, test.ready) {
				t.Fatalf("startup/readiness = %+v / %+v, want %+v",
					container.StartupProbe.ProbeHandler, container.ReadinessProbe.ProbeHandler, test.ready)
			}
			if !reflect.DeepEqual(container.LivenessProbe.ProbeHandler, test.liveness) {
				t.Fatalf("liveness = %+v, want %+v", container.LivenessProbe.ProbeHandler, test.liveness)
			}
			service := r.generateService(probedRouter(), test.mode)
			if !reflect.DeepEqual(service.Spec.Ports[0], test.service) {
				t.Fatalf("Service traffic port = %+v, want %+v", service.Spec.Ports[0], test.service)
			}
		})
	}
}

// The Helm chart and the Operator deploy the same Router. Their defaults must
// agree, so neither launcher knows a mode, image or port the other doesn't.
func TestOperatorDefaultsMatchTheHelmChart(t *testing.T) {
	raw, err := os.ReadFile("../../helm/semantic-router/values.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var values struct {
		Gateway struct {
			Mode string `json:"mode"`
		} `json:"gateway"`
		Image struct {
			Repository string `json:"repository"`
		} `json:"image"`
		Service struct {
			GRPC    struct{ Port int32 } `json:"grpc"`
			API     struct{ Port int32 } `json:"api"`
			Metrics struct{ Port int32 } `json:"metrics"`
		} `json:"service"`
	}
	if err = yaml.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	if values.Gateway.Mode != GatewayModeStandalone {
		t.Errorf("chart gateway.mode = %q, Operator default = %q", values.Gateway.Mode, GatewayModeStandalone)
	}
	repository, _, _ := strings.Cut(DefaultImage, ":")
	if values.Image.Repository != repository {
		t.Errorf("chart image.repository = %q, Operator image = %q", values.Image.Repository, repository)
	}
	ports := [3]int32{values.Service.GRPC.Port, values.Service.API.Port, values.Service.Metrics.Port}
	if ports != [3]int32{DefaultGRPCPort, DefaultAPIPort, DefaultMetricsPort} {
		t.Errorf("chart ext_proc, API and metrics ports = %v, Operator = %v", ports,
			[3]int32{DefaultGRPCPort, DefaultAPIPort, DefaultMetricsPort})
	}
	template, err := os.ReadFile("../../helm/semantic-router/templates/deployment.yaml")
	if err != nil {
		t.Fatal(err)
	}
	for _, arg := range routerGatewayArgs(GatewayModeStandalone)[1:] {
		if !strings.Contains(string(template), "- "+arg+"\n") {
			t.Errorf("the chart's standalone Router lacks the Operator's %s", arg)
		}
	}
	if !strings.Contains(string(template), `- -gateway={{ include "semantic-router.gatewayMode" . }}`) {
		t.Error("the chart must pass its gateway mode to the Router as the Operator does")
	}
}

func TestRouterReadinessProbeTransport(t *testing.T) {
	for _, test := range []struct {
		name string
		args []string
		grpc bool
	}{
		{name: "default", grpc: true},
		{name: "plaintext", args: []string{"--secure=false"}, grpc: true},
		{name: "TLS", args: []string{"--secure=true"}},
		{name: "bare TLS flag", args: []string{"-secure"}},
		{name: "double dash TLS flag", args: []string{"--secure"}},
		{name: "boolean alias", args: []string{"-secure=1"}},
		{name: "flag terminator", args: []string{"--", "--secure"}, grpc: true},
		{name: "last flag wins", args: []string{"--secure=true", "-secure=false"}, grpc: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			r := &SemanticRouterReconciler{}
			container := r.generateContainers(probedRouter(test.args...), GatewayModeIntegration)[0]
			want := corev1.ProbeHandler{TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt(50051)}}
			if test.grpc {
				want = corev1.ProbeHandler{GRPC: &corev1.GRPCAction{Port: 50051}}
			}
			for _, probe := range []*corev1.Probe{container.StartupProbe, container.ReadinessProbe} {
				if !reflect.DeepEqual(probe.ProbeHandler, want) {
					t.Fatalf("probe = %+v, want %+v", probe.ProbeHandler, want)
				}
			}
			if container.LivenessProbe.TCPSocket == nil {
				t.Fatal("ext_proc liveness must remain a TCP check")
			}
		})
	}
}

func TestRouterProbeSettingsArePreserved(t *testing.T) {
	r := &SemanticRouterReconciler{}
	for _, mode := range []string{GatewayModeStandalone, GatewayModeIntegration} {
		t.Run(mode, func(t *testing.T) {
			sr := &vllmv1alpha1.SemanticRouter{}
			container := r.generateContainers(sr, mode)[0]
			if container.StartupProbe != nil || container.ReadinessProbe != nil || container.LivenessProbe != nil {
				t.Fatal("omitted probes must remain unset")
			}
			sr = probedRouter()
			disabled := false
			sr.Spec.StartupProbe.Enabled = &disabled
			sr.Spec.ReadinessProbe.Enabled = &disabled
			sr.Spec.LivenessProbe.Enabled = &disabled
			container = r.generateContainers(sr, mode)[0]
			if container.StartupProbe != nil || container.ReadinessProbe != nil || container.LivenessProbe != nil {
				t.Fatal("disabled probes must remain unset")
			}
			period, timeout, failures, delay := int32(7), int32(4), int32(11), int32(19)
			settings := &vllmv1alpha1.ProbeSpec{
				PeriodSeconds: &period, TimeoutSeconds: &timeout,
				FailureThreshold: &failures, InitialDelaySeconds: &delay,
			}
			sr.Spec.StartupProbe, sr.Spec.ReadinessProbe, sr.Spec.LivenessProbe = settings, settings, settings
			container = r.generateContainers(sr, mode)[0]
			for _, probe := range []*corev1.Probe{container.StartupProbe, container.ReadinessProbe, container.LivenessProbe} {
				if probe.PeriodSeconds != period || probe.TimeoutSeconds != timeout || probe.FailureThreshold != failures {
					t.Fatalf("custom probe timing changed: %+v", probe)
				}
			}
			if container.ReadinessProbe.InitialDelaySeconds != delay || container.LivenessProbe.InitialDelaySeconds != delay {
				t.Fatal("custom initial delay changed")
			}
		})
	}
}
