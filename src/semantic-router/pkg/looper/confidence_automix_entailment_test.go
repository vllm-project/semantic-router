/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func newTestLooper() *ConfidenceLooper {
	return NewConfidenceLooper(&config.LooperConfig{})
}

func buildAutoMixEntailmentRequest(question string) *Request {
	params := openai.ChatCompletionNewParams{
		Model: "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(question),
		},
	}
	return &Request{
		OriginalRequest: &params,
		DecisionName:    "test_decision",
	}
}

func newStubVerifierServer(t *testing.T, confidence float64, shouldEscalate bool, verifiedCount, totalSamples int) (*httptest.Server, *int) {
	t.Helper()
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/verify" {
			http.NotFound(w, r)
			return
		}
		callCount++
		body := map[string]interface{}{
			"confidence":      confidence,
			"should_escalate": shouldEscalate,
			"verified_count":  verifiedCount,
			"total_samples":   totalSamples,
			"threshold":       0.7,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(body)
	}))
	return server, &callCount
}

func TestPerformAutoMixEntailment_AcceptedAtThreshold(t *testing.T) {
	server, callCount := newStubVerifierServer(t, 0.85, false, 5, 5)
	defer server.Close()
	t.Cleanup(automixVerifierCacheReset)

	looper := newTestLooper()
	evaluator := &ConfidenceEvaluator{
		Method:            MethodAutoMixEntailment,
		Threshold:         0.7,
		VerifierServerURL: server.URL,
	}
	req := buildAutoMixEntailmentRequest("What is 2+2?")

	conf, accepted, err := looper.performAutoMixEntailment(context.Background(), req, evaluator, "small-model", "4")
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if !accepted {
		t.Errorf("expected accepted=true (confidence=0.85 >= threshold=0.7), got false")
	}
	if conf < 0.84 || conf > 0.86 {
		t.Errorf("expected confidence ~0.85, got %v", conf)
	}
	if *callCount != 1 {
		t.Errorf("expected 1 verifier call, got %d", *callCount)
	}
}

func TestPerformAutoMixEntailment_RejectedBelowThreshold(t *testing.T) {
	server, _ := newStubVerifierServer(t, 0.4, true, 2, 5)
	defer server.Close()
	t.Cleanup(automixVerifierCacheReset)

	looper := newTestLooper()
	evaluator := &ConfidenceEvaluator{
		Method:            MethodAutoMixEntailment,
		Threshold:         0.7,
		VerifierServerURL: server.URL,
	}
	req := buildAutoMixEntailmentRequest("What is the capital of France?")

	conf, accepted, err := looper.performAutoMixEntailment(context.Background(), req, evaluator, "small-model", "Madrid")
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if accepted {
		t.Errorf("expected accepted=false (confidence=0.4 < threshold=0.7), got true")
	}
	if conf > 0.5 {
		t.Errorf("expected low confidence, got %v", conf)
	}
}

func TestPerformAutoMixEntailment_MissingVerifierURL(t *testing.T) {
	t.Cleanup(automixVerifierCacheReset)
	looper := newTestLooper()
	evaluator := &ConfidenceEvaluator{
		Method:    MethodAutoMixEntailment,
		Threshold: 0.7,
	}
	req := buildAutoMixEntailmentRequest("anything")

	_, _, err := looper.performAutoMixEntailment(context.Background(), req, evaluator, "small-model", "answer")
	if err == nil {
		t.Fatal("expected error when verifier_server_url is empty, got nil")
	}
	if !strings.Contains(err.Error(), "verifier_server_url") {
		t.Errorf("expected error mentioning verifier_server_url, got: %v", err)
	}
}

func TestPerformAutoMixEntailment_VerifierUnreachable(t *testing.T) {
	t.Cleanup(automixVerifierCacheReset)
	looper := newTestLooper()
	evaluator := &ConfidenceEvaluator{
		Method:                 MethodAutoMixEntailment,
		Threshold:              0.7,
		VerifierServerURL:      "http://127.0.0.1:1",
		VerifierTimeoutSeconds: 1,
	}
	req := buildAutoMixEntailmentRequest("anything")

	_, accepted, err := looper.performAutoMixEntailment(context.Background(), req, evaluator, "small-model", "answer")
	if err == nil {
		t.Fatal("expected error from unreachable verifier, got nil")
	}
	if accepted {
		t.Error("expected accepted=false on verifier failure")
	}
}

func TestPerformAutoMixEntailment_ClientCachedPerURL(t *testing.T) {
	server, _ := newStubVerifierServer(t, 0.9, false, 5, 5)
	defer server.Close()
	t.Cleanup(automixVerifierCacheReset)

	c1 := getAutoMixVerifierClient(server.URL, 0)
	c2 := getAutoMixVerifierClient(server.URL, 0)
	if c1 != c2 {
		t.Error("expected getAutoMixVerifierClient to return the same client for repeated (url, timeout) tuples")
	}

	c3 := getAutoMixVerifierClient(server.URL, 30)
	if c3 == c1 {
		t.Error("expected a distinct client when timeout differs")
	}
}

func automixVerifierCacheReset() {
	automixVerifierCache.Range(func(k, _ interface{}) bool {
		automixVerifierCache.Delete(k)
		return true
	})
}
