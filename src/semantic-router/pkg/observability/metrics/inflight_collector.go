/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
)

// inflightCollector exposes pkg/inflight as the prometheus gauge
// llm_model_inflight_requests at scrape time. Using a collector rather than a
// GaugeVec mirrored from inc/dec calls keeps pkg/inflight the single source
// of truth and avoids drift if a wire-in point panics or is missed.
type inflightCollector struct {
	desc *prometheus.Desc
}

func newInflightCollector() *inflightCollector {
	return &inflightCollector{
		desc: prometheus.NewDesc(
			"llm_model_inflight_requests",
			"Number of chat completion requests currently in flight per model, source: pkg/inflight",
			[]string{"model"},
			nil,
		),
	}
}

func (c *inflightCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.desc
}

func (c *inflightCollector) Collect(ch chan<- prometheus.Metric) {
	for model, count := range inflight.Snapshot() {
		ch <- prometheus.MustNewConstMetric(c.desc, prometheus.GaugeValue, float64(count), model)
	}
}

func init() {
	prometheus.MustRegister(newInflightCollector())
}
