package modelservice

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const ejectedReason = "replica is ejected after consecutive timeouts"

// Why an ejected replica returned to dispatch.
const (
	readmittedCooldown = "cooldown"
	readmittedAnswered = "answered"
)

// recordTimeouts counts the exchanges in a row that ran out of their deadline
// on w and ejects it at the deployment's threshold. A successful answer starts
// the count again and readmits an ejected replica at once. Canceled, rejected
// and failed exchanges neither count nor reset it. Ejection leaves the count,
// so the first timeout after a cool-down ejects the replica again.
func (p *replicaPool) recordTimeouts(w replicaWorker, outcome replicaOutcome, dispatched time.Time) {
	if outcome != replicaOK && outcome != replicaTimeout {
		return
	}
	now := time.Now()
	load := &w.member.served.load
	load.mu.Lock()
	readmitted := ""
	if load.expireLocked(now) {
		readmitted = readmittedCooldown
	}
	if outcome == replicaOK {
		if !load.ejected.IsZero() {
			readmitted = readmittedAnswered
		}
		load.timeouts, load.ejections, load.ejected, load.answered = 0, 0, time.Time{}, now
	} else {
		load.timeouts++
		if load.timeouts == 1 || dispatched.Before(load.streak) {
			load.streak = dispatched
		}
	}
	timeouts, out := load.timeouts, !load.ejected.IsZero()
	load.mu.Unlock()
	if readmitted != "" {
		p.readmitted(w, readmitted)
	}
	threshold := p.ejection.ConsecutiveTimeouts
	if outcome == replicaTimeout && !out && threshold > 0 && timeouts >= threshold {
		p.eject(w, timeouts)
	}
}

// eject takes w out of dispatch for its cool-down and asks its watcher for a
// probe. It holds back unless another ready replica answered after the first
// of w's timed-out exchanges was dispatched, so a model that is slow
// everywhere keeps every replica, and while the deployment's share of
// replicas is already ejected.
func (p *replicaPool) eject(w replicaWorker, timeouts int) {
	if len(p.workers) < 2 {
		return
	}
	p.manager.ejecting.Lock()
	defer p.manager.ejecting.Unlock()
	now := time.Now()
	p.mu.Lock()
	ready, ejected, _ := p.classifyLocked(now)
	p.mu.Unlock()
	load := &w.member.served.load
	load.mu.Lock()
	streak := load.streak
	load.mu.Unlock()
	answering, out := false, 0
	for _, other := range ready {
		if other.id != w.id && other.member.served.load.answeredAfter(streak) {
			answering = true
		}
	}
	for _, other := range ejected {
		if other.id != w.id {
			out++
		}
	}
	withheld := ""
	switch {
	case !answering:
		withheld = "no other ready replica answered since its timeouts began"
	case out >= max(1, len(p.workers)*p.ejection.MaxEjectionPercent/100):
		withheld = "max_ejection_percent of the replicas are ejected"
	}
	if withheld != "" {
		if timeouts == p.ejection.ConsecutiveTimeouts {
			logging.ComponentWarnEvent("model_runtime", "replica_ejection_withheld", map[string]interface{}{
				"deployment": p.name, "replica": w.id, "consecutive_timeouts": timeouts, "reason": withheld,
			})
		}
		return
	}
	load.mu.Lock()
	if load.timeouts < p.ejection.ConsecutiveTimeouts || !load.ejected.IsZero() {
		load.mu.Unlock()
		return
	}
	load.ejections++
	ejections := load.ejections
	cooldown := ejectionCooldown(p.ejection, ejections)
	load.ejected = now.Add(cooldown)
	load.mu.Unlock()
	replicaEjectionsTotal.WithLabelValues(p.name, w.id).Inc()
	logging.ComponentWarnEvent("model_runtime", "replica_ejected", map[string]interface{}{
		"deployment": p.name, "replica": w.id, "consecutive_timeouts": timeouts,
		"ejections_in_a_row": ejections, "cooldown_ms": cooldown.Milliseconds(),
	})
	w.member.group.requestProbe()
}

func (p *replicaPool) readmitted(w replicaWorker, reason string) {
	replicaReadmissionsTotal.WithLabelValues(p.name, w.id, reason).Inc()
	logging.ComponentEvent("model_runtime", "replica_readmitted", map[string]interface{}{
		"deployment": p.name, "replica": w.id, "reason": reason,
	})
}

// ejectionCooldown is how long the k-th ejection in a row lasts: the base,
// doubled for each earlier one, up to the maximum.
func ejectionCooldown(ejection config.ReplicaEjection, k int) time.Duration {
	cooldown := ejection.BaseEjection
	for i := 1; i < k && cooldown < ejection.MaxEjection; i++ {
		cooldown *= 2
	}
	return min(cooldown, ejection.MaxEjection)
}

// expireLocked returns a replica whose cool-down has ended to dispatch and
// reports whether it did.
func (l *replicaLoad) expireLocked(now time.Time) bool {
	if l.ejected.IsZero() || now.Before(l.ejected) {
		return false
	}
	l.ejected = time.Time{}
	return true
}

func (l *replicaLoad) answeredAfter(t time.Time) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.answered.After(t)
}
