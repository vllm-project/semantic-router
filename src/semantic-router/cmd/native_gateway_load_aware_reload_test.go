package main

import (
	"fmt"
	"net"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

func reloadLoadAware(t *testing.T, server interface {
	ConfigSnapshots() *configsnapshot.Manager
}, document string,
) *configsnapshot.Snapshot {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := server.ConfigSnapshots().Apply(t.Context(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceAPI}, Config: cfg, Document: []byte(document),
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func waitForTelemetryCleanup(t *testing.T, store *backendtelemetry.Store) {
	t.Helper()
	id := backendtelemetry.Identity{Model: "m", BackendRef: "gpu-a", Engine: backendtelemetry.EngineVLLM}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, held := store.Get(id); !held {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("drained telemetry generation retained its samples")
}

func TestNativeGatewayLoadAwareReloadOwnsItsReaderAndMetrics(t *testing.T) {
	a, b := newFakeVLLM(t, "gpu-a", 0), newFakeVLLM(t, "gpu-b", 50)
	server, front := newLoadAwareGateway(t, a, b)
	old, err := server.Pin()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if old != nil {
			old.Release()
		}
	})
	oldPart := old.Snapshot.Part(configsnapshot.ComponentBackendTelemetry).(*backendTelemetryRuntime)
	waitForQueueDepth(t, oldPart.store, "gpu-a", 0)
	waitForQueueDepth(t, oldPart.store, "gpu-b", 50)
	if counts := servedBy(t, front.URL, 20); counts["gpu-a"] != 20 {
		t.Fatalf("initial route = %v", counts)
	}

	// The serving endpoints and replica identities stay unchanged; only the
	// metrics sources move, and the old generation stays pinned while we warm.
	metricsA, metricsB := newFakeVLLM(t, "metrics-a", 50), newFakeVLLM(t, "metrics-b", 0)
	document := loadAwareDocument(a, b)
	for _, target := range []struct{ backend, metrics *fakeVLLM }{{a, metricsA}, {b, metricsB}} {
		endpoint := "endpoint: " + target.backend.hostPort()
		port := target.metrics.server.Listener.Addr().(*net.TCPAddr).Port
		document = strings.Replace(document, endpoint, endpoint+fmt.Sprintf("\n          metrics: {port: %d}", port), 1)
	}
	next := reloadLoadAware(t, server, document)
	nextPart := next.Part(configsnapshot.ComponentBackendTelemetry).(*backendTelemetryRuntime)
	if nextPart.store == oldPart.store {
		t.Fatal("a changed metrics source reused old samples")
	}
	oldSet := old.Snapshot.Part(configsnapshot.ComponentUpstream).(*upstream.Set)
	nextSet := next.Part(configsnapshot.ComponentUpstream).(*upstream.Set)
	if nextSet == oldSet || !reflect.DeepEqual(nextSet.Topology(), oldSet.Topology()) {
		t.Fatal("reload must rebind the reader while leaving reusable cluster specs unchanged")
	}
	waitForQueueDepth(t, nextPart.store, "gpu-a", 50)
	waitForQueueDepth(t, nextPart.store, "gpu-b", 0)
	if counts := servedBy(t, front.URL, 20); counts["gpu-b"] != 20 {
		t.Fatalf("new generation read old metrics: %v", counts)
	}
	waitForQueueDepth(t, oldPart.store, "gpu-a", 0)
	old.Release()
	old = nil
	waitForTelemetryCleanup(t, oldPart.store)
	waitForQueueDepth(t, nextPart.store, "gpu-b", 0)

	// Changing only TTL/poll settings also replaces the store and reader.
	settingsDocument := strings.Replace(strings.Replace(document, "poll_interval: 20ms", "poll_interval: 30ms", 1), "ttl: 200ms", "ttl: 300ms", 1)
	settings := reloadLoadAware(t, server, settingsDocument)
	settingsPart := settings.Part(configsnapshot.ComponentBackendTelemetry).(*backendTelemetryRuntime)
	if settingsPart.store == nextPart.store || settings.Part(configsnapshot.ComponentUpstream) == nextSet {
		t.Fatal("a settings-only reload retained an old telemetry reader")
	}
	waitForQueueDepth(t, settingsPart.store, "gpu-a", 50)
	waitForQueueDepth(t, settingsPart.store, "gpu-b", 0)
	if counts := servedBy(t, front.URL, 20); counts["gpu-b"] != 20 {
		t.Fatalf("settings reload lost telemetry selection: %v", counts)
	}

	// An unrelated setting preserves the complete telemetry and upstream parts.
	unchanged := reloadLoadAware(t, server, strings.Replace(settingsDocument, "global:", "global:\n  router:\n    list_backend_models: true", 1))
	if unchanged.Part(configsnapshot.ComponentBackendTelemetry) != settingsPart || unchanged.Part(configsnapshot.ComponentUpstream) != settings.Part(configsnapshot.ComponentUpstream) {
		t.Fatal("an unrelated reload rebuilt telemetry or upstream state")
	}
	disabled := reloadLoadAware(t, server, strings.Replace(settingsDocument, "enabled: true", "enabled: false", 1))
	if disabled.Part(configsnapshot.ComponentBackendTelemetry).(*backendTelemetryRuntime).store != nil {
		t.Fatal("disabled collection retained a reader")
	}
	if counts := servedBy(t, front.URL, 60); counts["gpu-a"] == 0 || counts["gpu-b"] == 0 {
		t.Fatalf("disabled telemetry did not fall back: %v", counts)
	}
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if strings.HasPrefix(family.GetName(), "llm_backend_telemetry_") {
			t.Fatalf("disabled generation retained metric series: %s", family.GetName())
		}
	}
	waitForTelemetryCleanup(t, settingsPart.store)
}
