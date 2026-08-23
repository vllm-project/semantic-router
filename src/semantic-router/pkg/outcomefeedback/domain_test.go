package outcomefeedback

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func TestDecodeRequestAcceptsBoundedModelOutcome(t *testing.T) {
	request, err := DecodeRequest([]byte(`{
  "replay_id":"replay-001",
  "target":"model",
  "target_ref":"model/primary",
  "target_revision":7,
  "verdict":"good_fit",
  "reason":"Matched the workload.",
  "score":0.875,
  "metadata":{"surface":"playground"}
}`))
	if err != nil {
		t.Fatal(err)
	}
	if request.ReplayID != "replay-001" || request.Target != TargetModel ||
		request.TargetRevision == nil || *request.TargetRevision != 7 ||
		request.Metadata["surface"] != "playground" {
		t.Fatalf("decoded request = %+v", request)
	}
}

func TestDecodeRequestRejectsCallerProvenanceAndUnboundedInput(t *testing.T) {
	tests := []struct {
		name    string
		payload []byte
	}{
		{
			name: "caller identity",
			payload: []byte(`{
  "replay_id":"replay-001","target":"route","verdict":"failed",
  "api_key_id":"00000000-0000-4000-8000-000000000001"
}`),
		},
		{name: "empty", payload: nil},
		{name: "trailing value", payload: []byte(`{"replay_id":"replay-001","target":"route","verdict":"failed"}{}`)},
		{name: "NUL", payload: []byte("{}\x00")},
		{name: "too large", payload: bytes.Repeat([]byte{'x'}, MaximumBodyBytes+1)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := DecodeRequest(test.payload); !errors.Is(err, ErrInvalid) {
				t.Fatalf("DecodeRequest() error = %v, want ErrInvalid", err)
			}
		})
	}
}

func TestRequestValidationBindsModelRevision(t *testing.T) {
	revision := int64(3)
	valid := Request{
		ReplayID: "replay-001", Target: TargetModel, TargetRef: "model-one",
		TargetRevision: &revision, Verdict: VerdictUnderpowered,
	}
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	withoutRevision := valid
	withoutRevision.TargetRevision = nil
	if err := withoutRevision.Validate(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("missing model revision error = %v", err)
	}
	routeWithRevision := valid
	routeWithRevision.Target = TargetRoute
	if err := routeWithRevision.Validate(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("route revision error = %v", err)
	}
}

func TestRequestDigestIsCanonicalAcrossMetadataOrder(t *testing.T) {
	left := Request{
		ReplayID: "replay-001", Target: TargetRoute, Verdict: VerdictGoodFit,
		Metadata: map[string]string{"z": "last", "a": "first"},
	}
	right := left
	right.Metadata = map[string]string{"a": "first", "z": "last"}
	leftDigest, err := RequestDigest(left)
	if err != nil {
		t.Fatal(err)
	}
	rightDigest, err := RequestDigest(right)
	if err != nil {
		t.Fatal(err)
	}
	if leftDigest != rightDigest {
		t.Fatalf("canonical request digests differ: %x != %x", leftDigest, rightDigest)
	}
}

func TestIdempotencyKeyBounds(t *testing.T) {
	for _, value := range []string{"", " leading", "trailing ", "contains\nnewline", strings.Repeat("x", MaximumIdempotencySize+1)} {
		if err := ValidateIdempotencyKey(value); !errors.Is(err, ErrInvalid) {
			t.Errorf("ValidateIdempotencyKey(%q) error = %v, want ErrInvalid", value, err)
		}
	}
	if err := ValidateIdempotencyKey("outcome-001"); err != nil {
		t.Fatal(err)
	}
}
