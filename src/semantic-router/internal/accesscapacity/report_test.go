package accesscapacity

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestReportCompletesOnlyWhenEveryInvariantPasses(t *testing.T) {
	config := DefaultConfig()
	report := NewReport(config, time.Date(2026, 8, 25, 1, 2, 3, 0, time.UTC), "redis")
	if report.Environment.Transport != "redis" {
		t.Fatalf("NewReport() transport = %q", report.Environment.Transport)
	}
	report.Projection = Projection{
		KeysPerSecond: 1000, MemoryBytesPerKey: 1024, IsolationSamples: 32,
	}
	report.Admission = Admission{
		Attempted: DefaultKeyCount, Allowed: DefaultKeyCount,
		Admission: Latency{Count: DefaultKeyCount, P99MS: 2}, MemoryBytesPerEvent: 512,
	}
	report.Usage = Usage{
		Produced:       DefaultKeyCount + int64(config.RequestLimit-1),
		Observed:       DefaultKeyCount + int64(config.RequestLimit-1),
		Acknowledged:   DefaultKeyCount + int64(config.RequestLimit-1),
		ObservationLag: Latency{Count: DefaultKeyCount + config.RequestLimit - 1, P99MS: 3},
	}
	report.Failover = Failover{GlobalQuotaStateConsistent: true}
	report.Complete(config)
	if report.Status != "passed" {
		t.Fatalf("Complete() status = %q, checks = %+v", report.Status, report.Checks)
	}
	markdown := report.Markdown()
	if !strings.Contains(markdown, "`router_replica`") || !strings.Contains(markdown, "not Router/Envoy HTTP E2E") {
		t.Fatalf("Markdown() lost scope boundary:\n%s", markdown)
	}

	report.Admission.Failed = 1
	report.Complete(config)
	if report.Status != "failed" {
		t.Fatal("Complete() passed a report with a failed admission")
	}
}

func TestReportErrorsExcludeConnectionDetails(t *testing.T) {
	report := Report{}
	appendReportError(&report, errors.New("connect to isolated Redis/Valkey: dial tcp 192.0.2.1:6379: refused"))
	if len(report.Errors) != 1 || report.Errors[0] != "connect to isolated Redis/Valkey" {
		t.Fatalf("reported errors = %v", report.Errors)
	}
}
