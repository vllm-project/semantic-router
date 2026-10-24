package pluginruntime

import (
	"testing"
)

func TestPluginsCannotWriteTheHeadersTheRouterOwns(t *testing.T) {
	request := NewPluginRequest("d", "m", nil)
	for _, name := range []string{
		":path", ":authority", "", "bad name", "content-length", "Host", "connection", "transfer-encoding",
		"x-selected-model", "X-Selected-Model", "x-vsr-selected-decision", "x-vsr-anything", "x-envoy-original-path",
		"x-envoy-upstream-rq-timeout-ms",
	} {
		if err := request.SetHeader(name, "v"); err == nil {
			t.Errorf("SetHeader(%q) succeeded", name)
		}
		if err := request.RemoveHeader(name); err == nil {
			t.Errorf("RemoveHeader(%q) succeeded", name)
		}
		if err := NewPluginResponse("d", "m", 200).SetHeader(name, "v"); err == nil {
			t.Errorf("response SetHeader(%q) succeeded", name)
		}
	}
	for _, name := range []string{"x-tenant", "Authorization", "x-request-tag"} {
		if err := request.SetHeader(name, "v"); err != nil {
			t.Errorf("SetHeader(%q) = %v", name, err)
		}
	}
	if got := request.Mutations(); len(got) != 3 || got[1].Name != "authorization" {
		t.Fatalf("mutations = %+v, want the three allowed headers in order, lowercased", got)
	}
}
