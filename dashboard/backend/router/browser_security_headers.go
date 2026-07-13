package router

import "net/http"

// withBrowserSecurityHeaders provides portable response hardening for the
// dashboard itself. HSTS remains the responsibility of the public TLS edge:
// the dashboard also supports intentional loopback HTTP development, and an
// application behind a proxy cannot prove the public hostname's TLS policy.
func withBrowserSecurityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("X-Frame-Options", "SAMEORIGIN")
		next.ServeHTTP(w, r)
	})
}
