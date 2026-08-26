package quotaruntime

import (
	"strings"
	"testing"
)

func TestScriptsUseServerTimeAndExactQuotaArithmetic(t *testing.T) {
	t.Parallel()

	if !strings.Contains(exactLua, `redis.call("TIME")`) {
		t.Fatal("exact runtime helper does not use Redis TIME")
	}
	if !strings.Contains(exactLua, "quota_limb_count = 6") ||
		!strings.Contains(exactLua, "quota_base = 10000000") {
		t.Fatal("exact runtime helper does not implement the six-limb quota domain")
	}
	if !strings.Contains(exactLua, "quota_multiply") {
		t.Fatal("exact runtime helper does not implement checked multiplication")
	}
	if strings.Contains(exactLua, "tonumber(value)") {
		t.Fatal("exact runtime helper converts a complete quota quantity to Lua floating point")
	}

	tests := []struct {
		name   string
		script string
	}{
		{name: "admit", script: admitLua},
		{name: "heartbeat", script: heartbeatLua},
		{name: "check access", script: checkAccessLua},
		{name: "journal dispatch", script: journalDispatchLua},
		{name: "begin dispatch", script: beginDispatchLua},
		{name: "begin attempt", script: beginAttemptLua},
		{name: "finish attempt", script: finishAttemptLua},
		{name: "read attempt evidence", script: readAttemptEvidenceLua},
		{name: "finalize", script: finalizeLua},
		{name: "release concurrency", script: releaseConcurrencyLua},
		{name: "read meters", script: readMetersLua},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if !strings.Contains(test.script, "quota_time_milliseconds()") &&
				!strings.Contains(test.script, "quota_server_time()") {
				t.Fatal("operation does not source its time from Redis")
			}
		})
	}
}

func TestAdmissionHeartbeatCannotRetargetOrReviveWork(t *testing.T) {
	t.Parallel()

	for _, contract := range []string{
		`admission_digest`,
		`plan_digest`,
		`lease_ms`,
		`concurrency_count`,
		`concurrency fingerprint differs`,
		`expired admission cannot be renewed`,
		`admission_plan_digest`,
		`redis.call("ZADD", KEYS[1], new_deadline, admission_id)`,
	} {
		if !strings.Contains(heartbeatLua, contract) {
			t.Fatalf("admission heartbeat is missing contract %q", contract)
		}
	}
	if !strings.Contains(finalizeLua, `"admission_plan_digest", admission_plan_digest`) {
		t.Fatal("terminal settlement does not retain the admission plan identity for a racing heartbeat")
	}
}

func TestAttemptJournalEnforcesSafeContiguousRetryAndAtomicCleanup(t *testing.T) {
	t.Parallel()

	for _, contract := range []string{
		`dispatch journal differs`,
		`dispatch deadline is outside admission lease`,
		`attempt_count`,
	} {
		if !strings.Contains(beginDispatchLua, contract) {
			t.Fatalf("begin-dispatch script is missing contract %q", contract)
		}
	}
	for _, contract := range []string{
		`attempt_number ~= attempt_count + 1`,
		`only known-zero evidence permits retry`,
		`attempt was already started`,
	} {
		if !strings.Contains(beginAttemptLua, contract) {
			t.Fatalf("begin-attempt script is missing contract %q", contract)
		}
	}
	if !strings.Contains(finishAttemptLua, `attempt evidence differs`) ||
		!strings.Contains(finishAttemptLua, `current_state ~= "started"`) {
		t.Fatal("finish-attempt script does not provide idempotent conflict detection")
	}
	if !strings.Contains(readAttemptEvidenceLua, `state`) ||
		!strings.Contains(readAttemptEvidenceLua, `completed_at`) ||
		!strings.Contains(readAttemptEvidenceLua, `revision`) {
		t.Fatal("attempt evidence read omits terminal state or timestamps")
	}
	for name, script := range map[string]string{
		"begin dispatch": beginDispatchLua,
		"begin attempt":  beginAttemptLua,
		"finish attempt": finishAttemptLua,
	} {
		if !strings.Contains(script, `HINCRBY`) || !strings.Contains(script, `revision`) {
			t.Fatalf("%s does not advance the admission attempt-evidence revision", name)
		}
	}
	if !strings.Contains(finalizeLua, `QUOTA_EVIDENCE_CHANGED`) ||
		!strings.Contains(finalizeLua, `expected_evidence_revision`) {
		t.Fatal("finalization does not compare-and-set authoritative attempt evidence")
	}
	if !strings.Contains(finalizeLua, `redis.call("DEL", KEYS[#KEYS])`) {
		t.Fatal("finalization does not atomically clean request attempt evidence")
	}
}

func TestFinalizationAtomicallyClassifiesEveryActualCounter(t *testing.T) {
	t.Parallel()

	for _, contract := range []string{
		`state == "known"`,
		`state == "unknown"`,
		`redis.call("XADD"`,
		`redis.call("SADD"`,
		`redis.call("ZREM"`,
		`finalization_digest`,
		`plan_digest`,
	} {
		if !strings.Contains(finalizeLua, contract) {
			t.Fatalf("finalization script is missing contract %q", contract)
		}
	}
}

func TestFinalizationAllowsAFullCrossingDebit(t *testing.T) {
	t.Parallel()

	if !strings.Contains(finalizeLua, "quota_add(update.used, amount)") {
		t.Fatal("finalization does not apply the full exact actual debit")
	}
	if strings.Contains(finalizeLua, "rate_limited") || strings.Contains(finalizeLua, "limit exhausted") {
		t.Fatal("finalization contains a pre-debit limit rejection")
	}
}

func TestAdmissionChecksAccessBeforeCounterMutation(t *testing.T) {
	t.Parallel()

	guard := strings.Index(admitLua, `quota_check_access`)
	requestCounter := strings.Index(admitLua, `quota_prune_request`)
	consume := strings.Index(admitLua, `redis.call("ZADD", event_key, now, admission_id)`)
	if guard < 0 || requestCounter < 0 || consume < 0 || guard > requestCounter || guard > consume {
		t.Fatal("access projection assertions are not evaluated before quota inspection and consumption")
	}
}

func TestAdmissionAppliesUsageBackpressureBeforeCounterMutation(t *testing.T) {
	t.Parallel()

	backlog := strings.Index(admitLua, `redis.call("XINFO", "GROUPS", usage_stream_key)`)
	consume := strings.Index(admitLua, `redis.call("ZADD", event_key, now, admission_id)`)
	if backlog < 0 || consume < 0 || backlog > consume {
		t.Fatal("usage backlog is not checked before quota consumption")
	}
	if strings.Contains(finalizeLua, `usage accounting backlog is full`) {
		t.Fatal("terminal finalization must remain available after admission")
	}
	if strings.Contains(admitLua, `redis.call("XLEN", usage_stream_key)`) {
		t.Fatal("historical acknowledged stream entries must not count as backlog")
	}
}

func TestAccessCheckHasNoQuotaOrPendingSideEffects(t *testing.T) {
	t.Parallel()

	if !strings.Contains(checkAccessLua, "quota_check_access") {
		t.Fatal("access check does not reuse the admission precondition evaluator")
	}
	for _, command := range []string{
		`redis.call("SET"`, `redis.call("HSET"`, `redis.call("ZADD"`,
		`redis.call("SADD"`, `redis.call("DEL"`, `redis.call("EXPIRE"`,
	} {
		if strings.Contains(checkAccessLua, command) {
			t.Fatalf("access check contains mutating command %s", command)
		}
	}
}

func TestScriptsBindTerminalMutationsToAdmissionDigests(t *testing.T) {
	t.Parallel()

	for name, script := range map[string]string{
		"dispatch":  journalDispatchLua,
		"finalize":  finalizeLua,
		"heartbeat": heartbeatLua,
		"release":   releaseConcurrencyLua,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if !strings.Contains(script, "admission_digest") {
				t.Fatal("script does not verify its admission digest")
			}
		})
	}
}
