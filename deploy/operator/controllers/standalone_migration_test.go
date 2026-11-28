package controllers

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	coreclient "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func TestStandaloneMigration(t *testing.T) {
	for _, interrupted := range []bool{false, true} {
		name := "old deployment"
		if interrupted {
			name = "already interrupted"
		}
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			r, sr := deploymentConfigTestRouter(t)
			sr.UID, sr.Generation = "router-uid", 11
			sr.Spec.Image.Repository, sr.Spec.Image.Tag = "test/router", "legacy"
			sr.Spec.ReadinessProbe = &vllmv1alpha1.ProbeSpec{Enabled: ptr.To(false)}
			sr.Spec.ImagePullSecrets = []corev1.LocalObjectReference{{Name: "private-registry"}}
			sr.Finalizers = []string{SemanticRouterFinalizer}
			sr.Status.Conditions = []metav1.Condition{{Type: "Available", Status: metav1.ConditionTrue}}
			dep := r.generateDeployment(sr, GatewayModeStandalone)
			dep.UID, dep.Generation = "workload-uid", 3
			dep.Status = appsv1.DeploymentStatus{Replicas: 1, ReadyReplicas: 1, UpdatedReplicas: 1, ObservedGeneration: 3}
			legacy := dep.Spec.Template.DeepCopy()
			legacy.Spec.Containers[0].Args = nil
			legacy.Spec.Containers = append(legacy.Spec.Containers, corev1.Container{
				Name:  "envoy-proxy",
				Ports: []corev1.ContainerPort{{Name: "envoy-http", ContainerPort: 8801}},
			})
			rs := &appsv1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "legacy", Namespace: sr.Namespace, Labels: semanticRouterLabels(sr)},
				Spec:       appsv1.ReplicaSetSpec{Template: *legacy.DeepCopy(), Replicas: ptr.To[int32](1)},
				Status:     appsv1.ReplicaSetStatus{Replicas: 1, ReadyReplicas: 1},
			}
			svc := r.generateService(sr, GatewayModeStandalone)
			if interrupted {
				dep.Status.Replicas = 2
				svc.Spec.Ports[0].TargetPort = intstr.FromString(DefaultListenerName)
			} else {
				dep.Spec.Template = *legacy.DeepCopy()
				svc.Spec.Ports[0].Name = "envoy-http"
			}
			if !interrupted {
				sr.Spec.Service.Type = corev1.ServiceTypeNodePort
				svc.Spec.Type = corev1.ServiceTypeNodePort
				svc.Spec.Ports[0].NodePort = 30801
			}
			previousPortName := svc.Spec.Ports[0].Name
			cm := &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: sr.Name + "-config", Namespace: sr.Namespace},
				Data:       map[string]string{"config.yaml": "original"},
			}
			for _, obj := range []client.Object{dep, svc, cm} {
				if err := controllerutil.SetControllerReference(sr, obj, r.Scheme); err != nil {
					t.Fatal(err)
				}
			}
			if err := controllerutil.SetControllerReference(dep, rs, r.Scheme); err != nil {
				t.Fatal(err)
			}
			r.Client = fake.NewClientBuilder().WithScheme(r.Scheme).WithStatusSubresource(sr).WithObjects(sr, dep, rs, svc, cm).Build()
			key := client.ObjectKeyFromObject(sr)
			reconcile := func() {
				t.Helper()
				if _, err := r.Reconcile(ctx, ctrl.Request{NamespacedName: key}); err != nil {
					t.Fatal(err)
				}
			}
			get := func(obj client.Object, k client.ObjectKey) {
				t.Helper()
				if err := r.Get(ctx, k, obj); err != nil {
					t.Fatal(err)
				}
			}
			check := &corev1.Pod{}
			probeKey := migrationProbeKey(sr)
			reconcile()
			get(check, probeKey)
			if len(check.Labels) != 0 || check.Spec.RestartPolicy != corev1.RestartPolicyNever ||
				!reflect.DeepEqual(check.Spec.ImagePullSecrets, sr.Spec.ImagePullSecrets) ||
				!reflect.DeepEqual(check.Spec.Containers[0].Args, []string{"-gateway=standalone", "-listener-address=0.0.0.0", "-help"}) {
				t.Fatalf("unexpected image check: %+v", check.Spec)
			}
			get(svc, key)
			if svc.Spec.Ports[0].TargetPort != intstr.FromInt(8801) || svc.Spec.Ports[0].Name != previousPortName {
				t.Fatal("legacy inference Service unreachable before image check")
			}
			check.Status.ContainerStatuses = []corev1.ContainerStatus{{
				Name: "check", State: corev1.ContainerState{Waiting: &corev1.ContainerStateWaiting{Reason: "ImagePullBackOff"}},
			}}
			if err := r.Status().Update(ctx, check); err != nil {
				t.Fatal(err)
			}
			reconcile()
			get(sr, key)
			progress := meta.FindStatusCondition(sr.Status.Conditions, "Progressing")
			if progress == nil || progress.Reason != "RouterImageCheckPending" || !strings.Contains(progress.Message, "ImagePullBackOff") {
				t.Fatalf("pull failure: %+v", progress)
			}
			finish := func(exit int32, message string) {
				t.Helper()
				get(check, probeKey)
				check.Status.Phase = corev1.PodSucceeded
				if exit != 0 {
					check.Status.Phase = corev1.PodFailed
				}
				check.Status.ContainerStatuses = []corev1.ContainerStatus{{
					Name: "check", State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: exit, Message: message}},
				}}
				if err := r.Status().Update(ctx, check); err != nil {
					t.Fatal(err)
				}
				reconcile()
			}
			finish(2, "flag provided but not defined: -gateway\nUsage of router:")
			reconcile()
			get(sr, key)
			progress = meta.FindStatusCondition(sr.Status.Conditions, "Progressing")
			if sr.Status.Phase != "Blocked" || !meta.IsStatusConditionTrue(sr.Status.Conditions, "Available") ||
				progress == nil || progress.Status != metav1.ConditionFalse || progress.Reason != incompatibleRouterImage ||
				progress.ObservedGeneration != sr.Generation {
				t.Fatalf("blocked status: %+v", sr.Status)
			}
			get(dep, key)
			if !reflect.DeepEqual(dep.Spec.Template, *legacy) {
				t.Fatal("expected the serving sidecar template")
			}
			get(cm, client.ObjectKeyFromObject(cm))
			if cm.Data["config.yaml"] != "original" {
				t.Fatal("overwrote the serving Router configuration")
			}
			sr.Spec.Image.Tag = "compatible"
			sr.Generation++
			if err := r.Update(ctx, sr); err != nil {
				t.Fatal(err)
			}
			reconcile()
			reconcile()
			get(check, probeKey)
			if check.Spec.Containers[0].Image != "test/router:compatible" {
				t.Fatal("checked the wrong image")
			}
			finish(0, "")
			get(dep, key)
			router := dep.Spec.Template.Spec.Containers[0]
			rolling := dep.Spec.Strategy.RollingUpdate
			if dep.Spec.MinReadySeconds == 0 || rolling == nil || rolling.MaxUnavailable == nil || rolling.MaxUnavailable.IntValue() != 0 {
				t.Fatalf("legacy replicas can retire before the replacement is available: %+v", dep.Spec.Strategy)
			}
			if len(dep.Spec.Template.Spec.Containers) != 1 || router.Image != "test/router:compatible" ||
				router.ReadinessProbe == nil || router.ReadinessProbe.HTTPGet == nil || router.ReadinessProbe.HTTPGet.Path != "/ready" {
				t.Fatalf("unsafe replacement: %+v", router)
			}
			if err := r.Delete(ctx, rs); err != nil {
				t.Fatal(err)
			}
			dep.Status = appsv1.DeploymentStatus{ObservedGeneration: dep.Generation, Replicas: 1, UpdatedReplicas: 1, ReadyReplicas: 1, AvailableReplicas: 1}
			if err := r.Status().Update(ctx, dep); err != nil {
				t.Fatal(err)
			}
			template := dep.Spec.Template.DeepCopy()
			reconcile()
			get(dep, key)
			get(sr, key)
			if sr.Status.Phase != "Running" || !reflect.DeepEqual(dep.Spec.Template, *template) {
				t.Fatalf("unexpected rollout after recovery: phase %s", sr.Status.Phase)
			}
			get(svc, key)
			if svc.Spec.Ports[0].Name != previousPortName || !interrupted && svc.Spec.Ports[0].NodePort != 30801 {
				t.Fatalf("inference port changed at cutover: %+v", svc.Spec.Ports[0])
			}
			if err := r.Get(ctx, probeKey, check); !apierrors.IsNotFound(err) {
				t.Fatalf("image check still exists: %v", err)
			}
		})
	}
}

func TestMigrationCheckExitCodeTwo(t *testing.T) {
	for _, tc := range []struct{ line, reason string }{
		{"flag provided but not defined: -listener-address", incompatibleRouterImage},
		{"entrypoint: configuration error", "RouterImageCheckFailed"},
	} {
		t.Run(tc.reason, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if req.URL.Path != "/api/v1/namespaces/default/pods/probe/log" || req.URL.Query().Get("container") != "check" {
					t.Errorf("unexpected log source %s", req.URL)
				}
				fmt.Fprintln(w, tc.line)
			}))
			defer server.Close()
			pods, err := coreclient.NewForConfig(&rest.Config{Host: server.URL})
			if err != nil {
				t.Fatal(err)
			}
			r := &SemanticRouterReconciler{probeLogs: pods}
			pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "probe", Namespace: "default", CreationTimestamp: metav1.Now()}}
			pod.Status = corev1.PodStatus{
				Phase: corev1.PodFailed,
				ContainerStatuses: []corev1.ContainerStatus{{
					Name:  "check",
					State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 2, Message: "end of usage text"}},
				}},
			}
			got, approved, err := r.migrationProbeResult(t.Context(), pod, "pinned", metav1.Condition{Status: metav1.ConditionTrue})
			if err != nil || approved || got.Reason != tc.reason {
				t.Fatalf("approved=%v, condition=%+v, err=%v", approved, got, err)
			}
		})
	}
}
