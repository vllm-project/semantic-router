package quotaruntime

import (
	_ "embed"

	"github.com/redis/go-redis/v9"
)

var (
	//go:embed scripts/exact.lua
	exactLua string

	//go:embed scripts/admit.lua
	admitLua string

	//go:embed scripts/heartbeat.lua
	heartbeatLua string

	//go:embed scripts/check_access.lua
	checkAccessLua string

	//go:embed scripts/journal_dispatch.lua
	journalDispatchLua string

	//go:embed scripts/begin_dispatch.lua
	beginDispatchLua string

	//go:embed scripts/begin_attempt.lua
	beginAttemptLua string

	//go:embed scripts/finish_attempt.lua
	finishAttemptLua string

	//go:embed scripts/read_attempt_evidence.lua
	readAttemptEvidenceLua string

	//go:embed scripts/next_expired.lua
	nextExpiredLua string

	//go:embed scripts/read_expired.lua
	readExpiredLua string

	//go:embed scripts/finalize.lua
	finalizeLua string

	//go:embed scripts/release_concurrency.lua
	releaseConcurrencyLua string

	//go:embed scripts/read_meters.lua
	readMetersLua string

	//go:embed scripts/reconcile_unknown.lua
	reconcileUnknownLua string

	//go:embed scripts/remove_reconciled_fence.lua
	removeReconciledFenceLua string
)

var (
	checkAccessScript           = redis.NewScript(exactLua + "\n" + checkAccessLua)
	admitScript                 = redis.NewScript(exactLua + "\n" + admitLua)
	heartbeatScript             = redis.NewScript(exactLua + "\n" + heartbeatLua)
	journalDispatchScript       = redis.NewScript(exactLua + "\n" + journalDispatchLua)
	beginDispatchScript         = redis.NewScript(exactLua + "\n" + beginDispatchLua)
	beginAttemptScript          = redis.NewScript(exactLua + "\n" + beginAttemptLua)
	finishAttemptScript         = redis.NewScript(exactLua + "\n" + finishAttemptLua)
	readAttemptEvidenceScript   = redis.NewScript(exactLua + "\n" + readAttemptEvidenceLua)
	nextExpiredScript           = redis.NewScript(exactLua + "\n" + nextExpiredLua)
	readExpiredScript           = redis.NewScript(exactLua + "\n" + readExpiredLua)
	finalizeScript              = redis.NewScript(exactLua + "\n" + finalizeLua)
	releaseConcurrencyScript    = redis.NewScript(exactLua + "\n" + releaseConcurrencyLua)
	readMetersScript            = redis.NewScript(exactLua + "\n" + readMetersLua)
	reconcileUnknownScript      = redis.NewScript(exactLua + "\n" + reconcileUnknownLua)
	removeReconciledFenceScript = redis.NewScript(exactLua + "\n" + removeReconciledFenceLua)
)
