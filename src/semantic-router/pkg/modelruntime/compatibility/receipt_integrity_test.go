package compatibility

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestReceiptGoldenRoundTrip(t *testing.T) {
	data, err := os.ReadFile("testdata/local-candle-cpu-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := ParseReceipt(data)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := receipt.CanonicalJSON()
	if err != nil {
		t.Fatal(err)
	}
	var golden bytes.Buffer
	if err = json.Compact(&golden, data); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(canonical, golden.Bytes()) {
		t.Fatalf("canonical bytes differ from golden fixture:\n%s", canonical)
	}
	const wantSubject = "sha256:3d4b1cabc20d54f51709a7127d7b6319f7fd6b1466d8e6fb92bbbb197ada08a2"
	const wantReceipt = "sha256:d02895afb86cb4ffeb146f0902f14048410cf70909612ab0da5bc097e1870c31"
	if receipt.SubjectDigest != wantSubject {
		t.Fatalf("subject digest = %s, want %s", receipt.SubjectDigest, wantSubject)
	}
	if err = receipt.VerifyDigest(wantReceipt); err != nil {
		t.Fatal(err)
	}
	roundTrip, err := ParseReceipt(canonical)
	if err != nil {
		t.Fatal(err)
	}
	again, err := roundTrip.CanonicalJSON()
	if err != nil || !bytes.Equal(again, canonical) {
		t.Fatalf("round-trip changed canonical bytes: %v", err)
	}
	if err = roundTrip.VerifyDigest(wantReceipt); err != nil {
		t.Fatal(err)
	}
}

func TestReceiptDigestDetectsTampering(t *testing.T) {
	baseline, err := NewReceipt(testSubject(), testSuiteDigest(), testChecks())
	if err != nil {
		t.Fatal(err)
	}
	expected, err := baseline.Digest()
	if err != nil {
		t.Fatal(err)
	}
	mutations := map[string]func(*Receipt){
		"subject": func(r *Receipt) { r.Subject.Precision = "float16" },
		"rehash subject": func(r *Receipt) {
			r.Subject.RouterRevision = "changed-revision"
			r.SubjectDigest, err = r.Subject.Digest()
			if err != nil {
				t.Fatal(err)
			}
		},
		"subject digest":    func(r *Receipt) { r.SubjectDigest = testSuiteDigest() },
		"suite digest":      func(r *Receipt) { r.QualificationSuiteDigest = "sha256:" + strings.Repeat("c", 64) },
		"outcome":           func(r *Receipt) { r.Checks[0].Passed = false },
		"details":           func(r *Receipt) { r.Checks[0].Details = "different evidence" },
		"check name":        func(r *Receipt) { r.Checks[0].Name = "different_check" },
		"check order":       func(r *Receipt) { r.Checks[0], r.Checks[1] = r.Checks[1], r.Checks[0] },
		"predicate type":    func(r *Receipt) { r.PredicateType = "different" },
		"predicate version": func(r *Receipt) { r.PredicateVersion = "v2" },
		"receipt schema":    func(r *Receipt) { r.SchemaVersion = "v2" },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			r := baseline
			r.Checks = append([]CheckOutcome(nil), baseline.Checks...)
			mutate(&r)
			data, marshalErr := json.Marshal(r)
			if marshalErr != nil {
				t.Fatal(marshalErr)
			}
			parsed, parseErr := ParseReceipt(data)
			if parseErr == nil && parsed.VerifyDigest(expected) == nil {
				t.Fatal("tampered evidence verified against the original digest")
			}
		})
	}
	baseline.Checks[0].Passed = false
	digest, err := baseline.Digest()
	if err != nil || digest == expected {
		t.Fatalf("failed evidence must have a distinct digest: %s, %v", digest, err)
	}
	if err = baseline.VerifyDigest(digest); err != nil {
		t.Fatalf("intact failed evidence must verify: %v", err)
	}
	if err = baseline.VerifyDigest("invalid"); err == nil {
		t.Fatal("malformed expected digest accepted")
	}
}

func TestReceiptCanonicalEncodingIgnoresJSONFormatting(t *testing.T) {
	receipt, err := NewReceipt(testSubject(), testSuiteDigest(), testChecks())
	if err != nil {
		t.Fatal(err)
	}
	receipt.Checks[0].Details = "<test> & 中文"
	canonical, err := receipt.CanonicalJSON()
	if err != nil || !bytes.Contains(canonical, []byte(receipt.Checks[0].Details)) {
		t.Fatalf("canonical encoding must preserve HTML and Unicode: %s, %v", canonical, err)
	}
	formatted, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseReceipt(formatted)
	if err != nil {
		t.Fatal(err)
	}
	again, err := parsed.CanonicalJSON()
	if err != nil || !bytes.Equal(canonical, again) {
		t.Fatalf("JSON formatting changed canonical bytes: %v", err)
	}
}

func TestReceiptSchemaIsConnectorNeutral(t *testing.T) {
	subject := testSubject()
	original, err := subject.Digest()
	if err != nil {
		t.Fatal(err)
	}
	// A schema-only vector, not a remote transport implementation or support claim.
	subject.Connector = "synthetic.remote.connector.v1"
	receipt, err := NewReceipt(subject, testSuiteDigest(), testChecks())
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Subject.SchemaVersion != SubjectSchemaVersionV1 || receipt.SubjectDigest == original {
		t.Fatal("another connector must retain the schema but change the identity")
	}
	if err = validateLocalCandleCPUSubject(subject); err == nil {
		t.Fatal("Candle runner must not accept another connector")
	}
}
