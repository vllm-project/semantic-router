package proxy

import (
	"net/http"
	"testing"
)

func TestStripDashboardAuthQueryPreservesUnrelatedRawQuery(t *testing.T) {
	raw := "theme=dark&auth%54oken=first&empty=&authToken=second&theme=light&bad%ZZ=value"
	want := "theme=dark&empty=&theme=light&bad%ZZ=value"
	if got := stripDashboardAuthQuery(raw); got != want {
		t.Fatalf("stripDashboardAuthQuery() = %q, want %q", got, want)
	}
}

func TestStripDashboardSessionCookiesPreservesUpstreamCookies(t *testing.T) {
	header := http.Header{}
	header.Add("Cookie", "vsr_session=first; grafana_session=g")
	header.Add("Cookie", "theme=d; vsr_session=second")

	stripDashboardSessionCookies(header)

	values := header.Values("Cookie")
	if len(values) != 2 || values[0] != "grafana_session=g" || values[1] != "theme=d" {
		t.Fatalf("Cookie headers = %#v", values)
	}
}

func TestStripDashboardSessionSetCookiesPreservesUpstreamCookies(t *testing.T) {
	header := http.Header{}
	header.Add("Set-Cookie", "vsr_session=attacker; Path=/; HttpOnly")
	header.Add("Set-Cookie", "grafana_session=g; Path=/embedded/grafana; HttpOnly")

	stripDashboardSessionSetCookies(header)

	values := header.Values("Set-Cookie")
	if len(values) != 1 || values[0] != "grafana_session=g; Path=/embedded/grafana; HttpOnly" {
		t.Fatalf("Set-Cookie headers = %#v", values)
	}
}
