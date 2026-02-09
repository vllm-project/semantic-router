package metrics

// RecordImageGenRequest records an image generation request
func RecordImageGenRequest(backend string, status string, latency float64) {
	ImageGenRequests.WithLabelValues(backend, status).Inc()
	if latency > 0 {
		ImageGenLatency.WithLabelValues(backend).Observe(latency)
	}
}
