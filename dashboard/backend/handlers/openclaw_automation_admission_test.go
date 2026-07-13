package handlers

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func waitForAutomationTestCondition(t *testing.T, description string, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", description)
}

func TestRoomAutomationAdmissionCoalescesNewestPerRoom(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	started := make(chan string, 2)
	release := make(chan struct{}, 2)
	h.roomAutomationProcess = func(_ string, messageID string) {
		started <- messageID
		<-release
	}

	if !h.enqueueRoomAutomation("room-a", "message-1") {
		t.Fatal("first message was not admitted")
	}
	if got := <-started; got != "message-1" {
		t.Fatalf("first message = %q", got)
	}
	if !h.enqueueRoomAutomation("room-a", "message-2") ||
		!h.enqueueRoomAutomation("room-a", "message-3") {
		t.Fatal("pending messages were not coalesced")
	}
	release <- struct{}{}
	if got := <-started; got != "message-3" {
		t.Fatalf("coalesced message = %q, want newest", got)
	}
	release <- struct{}{}

	waitForAutomationTestCondition(t, "room admission release", func() bool {
		h.roomAutomationAdmissionMu.Lock()
		defer h.roomAutomationAdmissionMu.Unlock()
		return len(h.roomAutomationAdmissions) == 0 && len(h.roomAutomationSlots) == 0
	})
}

func TestRoomAutomationAdmissionEnforcesGlobalLimitAndReleases(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	started := make(chan struct{}, maximumOpenClawAutomationWorkers)
	release := make(chan struct{})
	h.roomAutomationProcess = func(_, _ string) {
		started <- struct{}{}
		<-release
	}

	for index := 0; index < maximumOpenClawAutomationWorkers; index++ {
		if !h.enqueueRoomAutomation(fmt.Sprintf("room-%d", index), "message") {
			t.Fatalf("room %d was not admitted", index)
		}
	}
	for index := 0; index < maximumOpenClawAutomationWorkers; index++ {
		<-started
	}
	if h.enqueueRoomAutomation("room-overflow", "message") {
		t.Fatal("overflow room should be rejected")
	}
	close(release)
	waitForAutomationTestCondition(t, "all global admissions to release", func() bool {
		h.roomAutomationAdmissionMu.Lock()
		defer h.roomAutomationAdmissionMu.Unlock()
		return len(h.roomAutomationAdmissions) == 0 && len(h.roomAutomationSlots) == 0
	})
}

func TestRoomAutomationAdmissionPanicReleasesCapacity(t *testing.T) {
	h := newTestOpenClawHandler(t, t.TempDir(), false)
	panicked := make(chan struct{})
	var once sync.Once
	h.roomAutomationProcess = func(_, _ string) {
		once.Do(func() { close(panicked) })
		panic("secret payload must not be logged")
	}
	if !h.enqueueRoomAutomation("room-panic", "message") {
		t.Fatal("panic worker was not admitted")
	}
	<-panicked
	waitForAutomationTestCondition(t, "panic admission release", func() bool {
		h.roomAutomationAdmissionMu.Lock()
		defer h.roomAutomationAdmissionMu.Unlock()
		return len(h.roomAutomationAdmissions) == 0 && len(h.roomAutomationSlots) == 0
	})

	done := make(chan struct{})
	h.roomAutomationProcess = func(_, _ string) { close(done) }
	if !h.enqueueRoomAutomation("room-after-panic", "message") {
		t.Fatal("capacity was not reusable after panic")
	}
	<-done
}
