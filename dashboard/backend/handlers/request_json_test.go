package handlers

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDecodeBoundedJSONContract(t *testing.T) {
	type payload struct {
		Name string `json:"name"`
	}

	tests := []struct {
		name       string
		body       []byte
		limit      int64
		wantStatus int
		wantName   string
	}{
		{name: "valid", body: []byte(`{"name":"router"}`), limit: 64, wantName: "router"},
		{name: "empty", body: nil, limit: 64, wantStatus: http.StatusBadRequest},
		{name: "unknown field", body: []byte(`{"name":"router","extra":true}`), limit: 64, wantStatus: http.StatusBadRequest},
		{name: "trailing value", body: []byte(`{"name":"router"}{}`), limit: 64, wantStatus: http.StatusBadRequest},
		{name: "invalid unicode", body: []byte{'{', '"', 'n', 'a', 'm', 'e', '"', ':', '"', 0xff, '"', '}'}, limit: 64, wantStatus: http.StatusBadRequest},
		{name: "oversized", body: []byte(strings.Repeat(" ", 65)), limit: 64, wantStatus: http.StatusRequestEntityTooLarge},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(test.body))
			var got payload
			status, err := decodeBoundedJSON(recorder, request, test.limit, &got)
			if test.wantStatus == 0 {
				if err != nil || status != 0 {
					t.Fatalf("decodeBoundedJSON() = status %d, err %v", status, err)
				}
				if got.Name != test.wantName {
					t.Fatalf("decoded name = %q, want %q", got.Name, test.wantName)
				}
				return
			}
			if err == nil || status != test.wantStatus {
				t.Fatalf("decodeBoundedJSON() = status %d, err %v; want status %d", status, err, test.wantStatus)
			}
		})
	}
}
