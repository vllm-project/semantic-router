package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// NewServeMux returns a mux serving only the Prometheus scrape endpoint.
//
// The metrics listener deliberately avoids http.DefaultServeMux: packages such
// as net/http/pprof register handlers there from their init functions, which
// would otherwise expose debug endpoints on the metrics port.
func NewServeMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	return mux
}
