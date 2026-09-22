package controllers

import (
	"testing"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func TestCanonicalPIITokenWindowPreserved(t *testing.T) {
	for _, window := range []*vllmv1alpha1.PromptGuardWindowConfig{nil, {Size: 512, Overlap: 64}} {
		r := &SemanticRouterReconciler{}
		module, err := r.convertClassifierModule(&vllmv1alpha1.ClassifierConfig{
			PIIModel: &vllmv1alpha1.PIIModelConfig{
				UseMmBERT32K: true, MaxSequenceLength: 32768, Window: window,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		got := module.PII.PIIModel
		if !got.UseMmBERT32K || got.MaxSequenceLength != 32768 {
			t.Fatalf("lost local selector or total input budget: %+v", got)
		}
		if window == nil {
			if got.Window != nil {
				t.Fatal("translation injected an omitted window")
			}
		} else if got.Window == nil || got.Window.Size != window.Size || got.Window.Overlap != window.Overlap {
			t.Fatalf("lost window: %+v", got.Window)
		}
	}
}
