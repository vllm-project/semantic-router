package extproc

import (
	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

func ensureHeaderRemoved(mutation *ext_proc.HeaderMutation, key string) {
	for _, existing := range mutation.RemoveHeaders {
		if existing == key {
			return
		}
	}
	mutation.RemoveHeaders = append(mutation.RemoveHeaders, key)
}

func setHeaderValue(mutation *ext_proc.HeaderMutation, key, value string) {
	for _, option := range mutation.SetHeaders {
		if option.Header != nil && option.Header.Key == key {
			option.Header.RawValue = []byte(value)
			option.Header.Value = ""
			return
		}
	}
	mutation.SetHeaders = append(mutation.SetHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			RawValue: []byte(value),
		},
	})
}
