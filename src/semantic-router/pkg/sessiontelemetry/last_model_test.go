package sessiontelemetry

import (
	"testing"
	"time"
)

func TestRecordAndGetLastModel(t *testing.T) {
	ResetLastModelForTesting()
	RecordLastModel("sess-1", "model-a")
	got, ok := GetLastModel("sess-1")
	if !ok || got != "model-a" {
		t.Fatalf("GetLastModel = (%q,%v), want (model-a,true)", got, ok)
	}
}

func TestGetLastModelMissing(t *testing.T) {
	ResetLastModelForTesting()
	if got, ok := GetLastModel("nope"); ok || got != "" {
		t.Fatalf(`GetLastModel(missing) = (%q,%v), want ("",false)`, got, ok)
	}
}

func TestRecordLastModelOverwritesWithLatest(t *testing.T) {
	ResetLastModelForTesting()
	RecordLastModel("s", "a")
	RecordLastModel("s", "b")
	if got, _ := GetLastModel("s"); got != "b" {
		t.Fatalf("expected latest model b, got %q", got)
	}
}

func TestRecordLastModelIgnoresEmpty(t *testing.T) {
	ResetLastModelForTesting()
	RecordLastModel("", "a")
	RecordLastModel("s", "")
	if _, ok := GetLastModel(""); ok {
		t.Fatal("empty session id must not be stored")
	}
	if _, ok := GetLastModel("s"); ok {
		t.Fatal("empty model must not be stored")
	}
}

func TestGetLastModelExpiresAfterTTL(t *testing.T) {
	ResetLastModelForTesting()
	base := time.Now()
	setLastModelNowForTesting(func() time.Time { return base })
	defer setLastModelNowForTesting(nil)

	RecordLastModel("s", "a")
	setLastModelNowForTesting(func() time.Time { return base.Add(ttl + time.Minute) })
	if got, ok := GetLastModel("s"); ok || got != "" {
		t.Fatalf("expired entry should be gone, got (%q,%v)", got, ok)
	}
}

func TestRecordLastModelEnforcesSizeCap(t *testing.T) {
	ResetLastModelForTesting()
	base := time.Now()
	setLastModelNowForTesting(func() time.Time { return base }) // freeze: nothing TTL-expires
	defer setLastModelNowForTesting(nil)

	orig := maxLastModelSessions
	maxLastModelSessions = 3
	defer func() { maxLastModelSessions = orig }()

	for _, id := range []string{"s1", "s2", "s3", "s4", "s5"} {
		RecordLastModel(id, "m")
	}
	if n := lastModelSessionCount(); n > maxLastModelSessions {
		t.Fatalf("session count %d exceeds cap %d", n, maxLastModelSessions)
	}
}
