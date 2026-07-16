/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package selection

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestModelSelectionDurationMetricContract(t *testing.T) {
	InitializeMetrics()
	RecordSelection("static", "contract-decision", "contract-test-model", TierSupported, 0.9)

	count, err := testutil.GatherAndCount(prometheus.DefaultGatherer, "llm_model_selection_duration_seconds")
	if err != nil {
		t.Fatalf("gather llm_model_selection_duration_seconds: %v", err)
	}
	if count == 0 {
		t.Fatal("expected Prometheus metric family llm_model_selection_duration_seconds to be gathered")
	}
}
