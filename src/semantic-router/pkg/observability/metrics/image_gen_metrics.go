package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// ImageGenRequests tracks image generation requests
	ImageGenRequests = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "image_gen_requests_total",
			Help: "Total number of image generation requests",
		},
		[]string{"backend", "status"},
	)

	// ImageGenLatency tracks image generation latency
	ImageGenLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "image_gen_latency_seconds",
			Help:    "Image generation latency in seconds",
			Buckets: []float64{1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0},
		},
		[]string{"backend"},
	)
)

// RecordImageGenRequest records an image generation request
func RecordImageGenRequest(backend string, status string, latency float64) {
	ImageGenRequests.WithLabelValues(backend, status).Inc()
	if latency > 0 {
		ImageGenLatency.WithLabelValues(backend).Observe(latency)
	}
}
