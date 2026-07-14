package router

import "net/http"

const dashboardContentSecurityPolicy = "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'self'; form-action 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; frame-src 'self'; worker-src 'none'; manifest-src 'self'"

const dashboardPermissionsPolicy = "accelerometer=(), autoplay=(), bluetooth=(), browsing-topics=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), serial=(), usb=()"

// withBrowserSecurityHeaders provides portable response hardening for the
// dashboard itself. HSTS remains the responsibility of the public TLS edge:
// the dashboard also supports intentional loopback HTTP development, and an
// application behind a proxy cannot prove the public hostname's TLS policy.
func withBrowserSecurityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("X-Frame-Options", "SAMEORIGIN")
		w.Header().Set("Content-Security-Policy", dashboardContentSecurityPolicy)
		w.Header().Set("Permissions-Policy", dashboardPermissionsPolicy)
		next.ServeHTTP(w, r)
	})
}
