package proxy

import (
	"log"
	"net/http"
	"net/url"
)

// logProxyEvent deliberately excludes headers, raw queries, target userinfo,
// and error strings. Those values can contain browser or upstream credentials.
func logProxyEvent(event string, request *http.Request, target *url.URL, path string) {
	method := ""
	if request != nil {
		method = request.Method
		if path == "" && request.URL != nil {
			path = request.URL.EscapedPath()
		}
	}
	log.Printf(
		"%s method=%q target=%q path=%q",
		event,
		method,
		proxyTargetAuthority(target),
		path,
	)
}

func proxyTargetAuthority(target *url.URL) string {
	if target == nil || target.Scheme == "" || target.Host == "" {
		return ""
	}
	// url.URL.Host never contains Userinfo. Target construction rejects Userinfo
	// as well, so this representation is safe for operational diagnostics.
	return target.Scheme + "://" + target.Host
}
