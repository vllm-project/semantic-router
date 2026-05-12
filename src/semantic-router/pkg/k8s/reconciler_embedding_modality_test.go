package k8s

import (
	"context"
	"errors"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestReconcileEmbeddingModalityValidation drives a real Reconciler instance
// through validateAndUpdate using controller-runtime's fake client, verifying
// that the embedding-modality contract is enforced on the K8s reconcile path
// (the gap rootfs identified in PR #1880's review).
//
// This test must not pass if the validation block in
// reconciler.go::validateAndUpdate is removed; that property is what makes
// it the load-bearing artifact for the contract this PR adds.
//
// TODO: once an embedding-modality e2e profile lands (#1881), consider
// migrating to the fixture-driven pattern used by
// dynamic_config_regression_test.go.
func TestReconcileEmbeddingModalityValidation(t *testing.T) {
	cases := []reconcileEmbeddingModalityCase{
		{
			name:              "AudioRejected",
			queryModality:     "audio",
			ruleName:          "audio_rule_under_test",
			baseModelType:     "multimodal",
			wantValidationErr: true,
			errSubstrings:     []string{"audio_rule_under_test", "audio FFI", "planned"},
		},
		{
			name:              "ImageWithoutMultimodalRejected",
			queryModality:     "image",
			ruleName:          "image_rule_under_test",
			baseModelType:     "qwen3",
			wantValidationErr: true,
			errSubstrings:     []string{"image_rule_under_test", "model_type=multimodal"},
		},
		{
			name:              "ImageWithMultimodalAccepted",
			queryModality:     "image",
			ruleName:          "image_rule_under_test",
			baseModelType:     "multimodal",
			wantValidationErr: false,
		},
		{
			name:              "TextOnlyAccepted",
			queryModality:     "text",
			ruleName:          "text_rule_under_test",
			baseModelType:     "qwen3",
			wantValidationErr: false,
		},
		{
			// The legacy backward-compat path: rules authored before #1880
			// omit queryModality entirely. EffectiveQueryModality resolves
			// "" to text downstream, so the validator must accept this
			// regardless of the configured embedding model_type. Regression
			// canary for every IntelligentRoute that predates the field.
			name:              "OmittedModalityAccepted",
			queryModality:     "",
			ruleName:          "omitted_modality_rule_under_test",
			baseModelType:     "qwen3",
			wantValidationErr: false,
		},
		{
			name:              "UnknownModalityRejected",
			queryModality:     "imag",
			ruleName:          "unknown_modality_rule_under_test",
			baseModelType:     "multimodal",
			wantValidationErr: true,
			errSubstrings:     []string{"unknown_modality_rule_under_test", "unknown query_modality"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runReconcileEmbeddingModalityCase(t, tc)
		})
	}
}

// reconcileEmbeddingModalityCase is one row in the embedding-modality
// reconcile-path validation table. ruleName is named explicitly per case
// so the OmittedModalityAccepted case (queryModality = "") still has a
// meaningful rule name and so the per-case names cannot drift away from
// the substrings the assertion table expects.
type reconcileEmbeddingModalityCase struct {
	name              string
	queryModality     string
	ruleName          string
	baseModelType     string
	wantValidationErr bool
	errSubstrings     []string
}

// runReconcileEmbeddingModalityCase builds a Reconciler with a fake K8s
// client, runs validateAndUpdate, and asserts the returned error and the
// status conditions on the reconciled pool and route.
func runReconcileEmbeddingModalityCase(t *testing.T, tc reconcileEmbeddingModalityCase) {
	t.Helper()

	const namespace = "default"
	pool := buildEmbeddingModalityPool(namespace)
	route := buildEmbeddingModalityRoute(namespace, tc.ruleName, tc.queryModality)
	staticConfig := buildEmbeddingModalityStaticConfig(tc.baseModelType)

	reconciler := buildEmbeddingModalityReconciler(t, namespace, staticConfig, pool, route)
	gotErr := reconciler.validateAndUpdate(context.Background(), pool, route)

	assertEmbeddingModalityResult(t, tc, gotErr)
	assertEmbeddingModalityStatus(t, reconciler, namespace, tc)
}

// buildEmbeddingModalityPool returns a minimal IntelligentPool sufficient
// for the converter and the downstream embedding-modality validation.
func buildEmbeddingModalityPool(namespace string) *v1alpha1.IntelligentPool {
	return &v1alpha1.IntelligentPool{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "embedding-modality-pool",
			Namespace: namespace,
		},
		Spec: v1alpha1.IntelligentPoolSpec{
			DefaultModel: "deepseek-v3",
			Models: []v1alpha1.ModelConfig{
				{Name: "deepseek-v3"},
			},
		},
	}
}

// buildEmbeddingModalityRoute returns a minimal IntelligentRoute carrying
// exactly one EmbeddingSignal with the requested QueryModality, plus a
// matching decision so the converter has a complete graph to translate.
func buildEmbeddingModalityRoute(namespace, ruleName, queryModality string) *v1alpha1.IntelligentRoute {
	return &v1alpha1.IntelligentRoute{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "embedding-modality-route",
			Namespace: namespace,
		},
		Spec: v1alpha1.IntelligentRouteSpec{
			Signals: v1alpha1.Signals{
				Embeddings: []v1alpha1.EmbeddingSignal{
					{
						Name:              ruleName,
						Threshold:         0.7,
						Candidates:        []string{"anchor phrase"},
						AggregationMethod: "max",
						QueryModality:     queryModality,
					},
				},
			},
			Decisions: []v1alpha1.Decision{
				{
					Name:     "decision_under_test",
					Priority: 100,
					Signals: v1alpha1.SignalCombination{
						Operator: "AND",
						Conditions: []v1alpha1.SignalCondition{
							{Type: "embedding", Name: ruleName},
						},
					},
					ModelRefs: []v1alpha1.ModelRef{
						{Model: "deepseek-v3"},
					},
				},
			},
		},
	}
}

// buildEmbeddingModalityStaticConfig returns the static base RouterConfig
// the operator would supply to the reconciler. ConfigSource is set to
// Kubernetes to mirror operator-mode bootstrap, which is the path where
// validateConfigStructure early-returns and PR-C closes the resulting gap.
func buildEmbeddingModalityStaticConfig(modelType string) *config.RouterConfig {
	return &config.RouterConfig{
		ConfigSource: config.ConfigSourceKubernetes,
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{
					ModelType: modelType,
				},
			},
		},
	}
}

// buildEmbeddingModalityReconciler wires a Reconciler around a fake K8s
// client preloaded with the pool and route under test. Status subresources
// for both CRDs are explicitly registered so updatePoolStatus and
// updateRouteStatus exercise the same Status().Update path the controller
// uses in production.
func buildEmbeddingModalityReconciler(
	t *testing.T,
	namespace string,
	staticConfig *config.RouterConfig,
	pool *v1alpha1.IntelligentPool,
	route *v1alpha1.IntelligentRoute,
) *Reconciler {
	t.Helper()

	scheme := runtime.NewScheme()
	if err := v1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&v1alpha1.IntelligentPool{}, &v1alpha1.IntelligentRoute{}).
		WithObjects(pool, route).
		Build()

	return &Reconciler{
		client:         fakeClient,
		scheme:         scheme,
		namespace:      namespace,
		converter:      NewCRDConverter(),
		staticConfig:   staticConfig,
		onConfigUpdate: func(*config.RouterConfig) error { return nil },
	}
}

// assertEmbeddingModalityResult applies the case's expectations to the
// validateAndUpdate return value. The accepted cases must return nil; the
// rejected cases must return a wrapped embedding-validation error whose
// message contains every named substring.
func assertEmbeddingModalityResult(t *testing.T, tc reconcileEmbeddingModalityCase, gotErr error) {
	t.Helper()

	if !tc.wantValidationErr {
		if gotErr != nil {
			t.Fatalf("validateAndUpdate: expected no error for %s + model_type=%s, got: %v",
				tc.queryModality, tc.baseModelType, gotErr)
		}
		return
	}

	if gotErr == nil {
		t.Fatalf("validateAndUpdate: expected error, got nil")
	}

	const wantPrefix = "embedding modality validation failed:"
	if !strings.HasPrefix(gotErr.Error(), wantPrefix) {
		t.Errorf("error should start with %q so callers can pattern-match the failure class, got: %s",
			wantPrefix, gotErr.Error())
	}

	if errors.Unwrap(gotErr) == nil {
		t.Error("error should wrap the underlying validator error via fmt.Errorf %w so errors.Unwrap can retrieve it")
	}

	for _, want := range tc.errSubstrings {
		if !strings.Contains(gotErr.Error(), want) {
			t.Errorf("error should contain %q, got: %s", want, gotErr.Error())
		}
	}
}

// assertEmbeddingModalityStatus re-fetches the reconciled pool and route
// from the fake client and asserts the Ready condition matches the expected
// admission outcome. Validation failures must mark both CRs Ready=False
// with Reason=ValidationFailed; accepted configs must mark both Ready=True.
func assertEmbeddingModalityStatus(
	t *testing.T,
	reconciler *Reconciler,
	namespace string,
	tc reconcileEmbeddingModalityCase,
) {
	t.Helper()

	wantStatus := metav1.ConditionTrue
	wantReason := "Ready"
	if tc.wantValidationErr {
		wantStatus = metav1.ConditionFalse
		wantReason = "ValidationFailed"
	}

	ctx := context.Background()
	var refetchedPool v1alpha1.IntelligentPool
	if err := reconciler.client.Get(ctx, client.ObjectKey{
		Namespace: namespace,
		Name:      "embedding-modality-pool",
	}, &refetchedPool); err != nil {
		t.Fatalf("Get pool after reconcile: %v", err)
	}
	assertReadyCondition(t, "pool", refetchedPool.Status.Conditions, wantStatus, wantReason)

	var refetchedRoute v1alpha1.IntelligentRoute
	if err := reconciler.client.Get(ctx, types.NamespacedName{
		Namespace: namespace,
		Name:      "embedding-modality-route",
	}, &refetchedRoute); err != nil {
		t.Fatalf("Get route after reconcile: %v", err)
	}
	assertReadyCondition(t, "route", refetchedRoute.Status.Conditions, wantStatus, wantReason)
}

// assertReadyCondition fails the test if the named CR's Ready condition
// does not match the expected status and reason.
func assertReadyCondition(
	t *testing.T,
	subject string,
	conditions []metav1.Condition,
	wantStatus metav1.ConditionStatus,
	wantReason string,
) {
	t.Helper()

	for _, c := range conditions {
		if c.Type != "Ready" {
			continue
		}
		if c.Status != wantStatus {
			t.Errorf("%s Ready condition status = %s, want %s", subject, c.Status, wantStatus)
		}
		if c.Reason != wantReason {
			t.Errorf("%s Ready condition reason = %q, want %q", subject, c.Reason, wantReason)
		}
		return
	}
	t.Errorf("%s has no Ready condition; want status=%s reason=%q", subject, wantStatus, wantReason)
}
