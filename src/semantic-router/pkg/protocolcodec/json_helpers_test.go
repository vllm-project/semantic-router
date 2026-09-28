package protocolcodec

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestNestedWireValuesRetainStrictValidation(t *testing.T) {
	tests := []struct {
		name     string
		provider bool
		body     string
		wantCode string
	}{
		{name: "client array", body: `[{"type":"text","text":"hello"}]`},
		{name: "client scalar", body: `"instructions"`},
		{name: "provider array", provider: true, body: `[{"type":"output_text","text":"hello"}]`},
		{name: "client duplicate nested field", body: `[{"type":"text","TYPE":"other"}]`, wantCode: "duplicate_json_field"},
		{name: "provider duplicate nested field", provider: true, body: `[{"type":"text","TYPE":"other"}]`, wantCode: "upstream_duplicate_json_field"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var value any
			var err error
			if test.provider {
				err = decodeProviderValue([]byte(test.body), &value, llmprotocol.DefaultPolicy())
			} else {
				err = decodeWireValue([]byte(test.body), &value, llmprotocol.DefaultPolicy())
			}
			if test.wantCode == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != test.wantCode {
				t.Fatalf("error = %v, want code %q", err, test.wantCode)
			}
		})
	}
}

func TestWireEnvelopesRemainObjectOnly(t *testing.T) {
	for _, test := range []struct {
		name     string
		provider bool
		wantCode string
	}{
		{name: "client", wantCode: "invalid_json"},
		{name: "provider", provider: true, wantCode: "invalid_upstream_json"},
	} {
		t.Run(test.name, func(t *testing.T) {
			var value any
			var err error
			if test.provider {
				err = decodeProviderWire([]byte(`[]`), &value, llmprotocol.DefaultPolicy())
			} else {
				err = decodeWire([]byte(`[]`), &value, llmprotocol.DefaultPolicy())
			}
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != test.wantCode {
				t.Fatalf("error = %v, want code %q", err, test.wantCode)
			}
		})
	}
}

func TestTypedWireValuesRejectCaseFoldedFieldNames(t *testing.T) {
	type nestedWire struct {
		Type string `json:"type"`
	}
	type envelopeWire struct {
		Model string       `json:"model"`
		Items []nestedWire `json:"items"`
	}
	tests := []struct {
		name     string
		provider bool
		body     string
		wantCode string
	}{
		{name: "client top level", body: `{"Model":"m","items":[]}`, wantCode: "invalid_json"},
		{name: "client nested array", body: `{"model":"m","items":[{"Type":"text"}]}`, wantCode: "invalid_json"},
		{name: "provider top level", provider: true, body: `{"Model":"m","items":[]}`, wantCode: "invalid_upstream_json"},
		{name: "provider nested array", provider: true, body: `{"model":"m","items":[{"Type":"text"}]}`, wantCode: "invalid_upstream_json"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var value envelopeWire
			var err error
			if test.provider {
				err = decodeProviderWire([]byte(test.body), &value, llmprotocol.DefaultPolicy())
			} else {
				err = decodeWire([]byte(test.body), &value, llmprotocol.DefaultPolicy())
			}
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != test.wantCode {
				t.Fatalf("error = %v, want code %q", err, test.wantCode)
			}
		})
	}
}
