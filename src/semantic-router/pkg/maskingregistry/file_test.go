package maskingregistry

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func testFile(t *testing.T) *File {
	t.Helper()
	f, e := OpenFile(filepath.Join(t.TempDir(), "registry"), "test", "key-id")
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { _ = f.Close() })
	return f
}

func testStore(t *testing.T, s Store) {
	t.Helper()
	issued, e := s.Prepare("scope", "pii:PERSON", []int64{0, math.MaxInt64})
	if e != nil || issued[0] {
		t.Fatalf("prepare: %v %v", issued, e)
	}
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			i, err := s.Allocate("scope", "pii:PERSON", "digest")
			if err != nil || i != 1 {
				t.Errorf("concurrent same value: %d %v", i, err)
			}
		}()
	}
	wg.Wait()
	i, e := s.Allocate("scope", "pii:PERSON", "second")
	if e != nil || i != 2 {
		t.Fatalf("distinct value: %d %v", i, e)
	}
	issued, e = s.Prepare("scope", "pii:PERSON", []int64{0, 1, 2})
	if e != nil || issued[0] || !issued[1] || !issued[2] {
		t.Fatalf("membership: %v %v", issued, e)
	}
	// A run of reserved indexes is skipped in one step.
	if _, e = s.Prepare("scope", "pii:PERSON", []int64{3, 4, 5}); e != nil {
		t.Fatal(e)
	}
	if i, e = s.Allocate("scope", "pii:PERSON", "after the run"); e != nil || i != 6 {
		t.Fatalf("after a reserved run: %d %v", i, e)
	}
	i, e = s.Allocate("other scope", "pii:PERSON", "digest")
	if e != nil || i != 0 {
		t.Fatalf("scope isolation: %d %v", i, e)
	}
	if e = s.Claim("pii:PERSON", "[PERSON_{index}]"); e != nil {
		t.Fatal(e)
	}
	if e = s.Claim("rule:keyword:person", "[PERSON_{index}]"); e == nil {
		t.Fatal("template ownership collision accepted")
	}
}
func TestFileTransactions(t *testing.T) { testStore(t, testFile(t)) }
func TestFileRestartAndTornTail(t *testing.T) {
	f := testFile(t)
	testStore(t, f)
	path := f.path
	_ = f.Close()
	tail, e := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if e != nil {
		t.Fatal(e)
	}
	// A whole frame header whose record was cut short: an append that never
	// finished.
	head := binary.BigEndian.AppendUint32(nil, 80)
	head = binary.BigEndian.AppendUint32(head, crc32.ChecksumIEEE(head))
	_, _ = tail.Write(append(head, '{'))
	_ = tail.Close()
	reopened, e := OpenFile(path, "test", "key-id")
	if e != nil {
		t.Fatal(e)
	}
	defer reopened.Close()
	i, e := reopened.Allocate("scope", "pii:PERSON", "digest")
	if e != nil || i != 1 {
		t.Fatalf("reopen: %d %v", i, e)
	}
	i, e = reopened.Allocate("scope", "pii:PERSON", "third")
	if e != nil || i != 7 {
		t.Fatalf("tail recovery: %d %v", i, e)
	}
}

func TestFileRefusesMissingMismatchedAndCorruptLogs(t *testing.T) {
	for _, kind := range []string{"missing", "empty", "key", "namespace", "corrupt", "length", "lock"} {
		t.Run(kind, func(t *testing.T) {
			f := testFile(t)
			path := f.path
			if kind == "lock" {
				wait := lockWait
				lockWait = 200 * time.Millisecond
				t.Cleanup(func() { lockWait = wait })
				if _, e := OpenFile(path, "test", "key-id"); e == nil {
					t.Fatal("process lock bypassed")
				}
				return
			}
			_ = f.Close()
			name, key := "test", "key-id"
			switch kind {
			case "missing":
				_ = os.Remove(path)
			case "empty":
				_ = os.WriteFile(path, nil, 0o600)
			case "key":
				key = "different"
			case "namespace":
				name = "different"
			case "corrupt":
				b, _ := os.ReadFile(path)
				b[len(b)-1] ^= 1
				_ = os.WriteFile(path, b, 0o600)
			case "length":
				// A larger length on a committed allocation reads past the end
				// of the log, like an append that never finished, and cutting
				// the log there would hand its index to another value.
				reopened, e := OpenFile(path, name, key)
				if e != nil {
					t.Fatal(e)
				}
				if _, e = reopened.Allocate("s", "t", "d"); e != nil {
					t.Fatal(e)
				}
				_ = reopened.Close()
				b, _ := os.ReadFile(path)
				second := 8 + int(binary.BigEndian.Uint32(b)) + 32
				b[second+2] ^= 0x01
				_ = os.WriteFile(path, b, 0o600)
			}
			if opened, e := OpenFile(path, name, key); e == nil {
				opened.Close()
				t.Fatal("replacement accepted")
			}
		})
	}
}

func TestFileDisappearanceBeforeCommit(t *testing.T) {
	f := testFile(t)
	_ = os.Remove(f.path)
	if _, e := f.Allocate("s", "t", "d"); e == nil {
		t.Fatal("dispatch could use lost allocation")
	}
}

func TestIdentityDigests(t *testing.T) {
	t.Setenv("MASKING_TEST_KEY", fmt.Sprintf("%064x", 1))
	ns := config.MaskingIdentityNamespace{KeyEnv: "MASKING_TEST_KEY", Scope: config.MaskingIdentityScope, Registry: config.MaskingRegistryConfig{Backend: "file", Path: filepath.Join(t.TempDir(), "log")}}
	id, e := Open("test", ns)
	if e != nil {
		t.Fatal(e)
	}
	a, _ := id.Scope("principal-A", "")
	b, _ := id.Scope("principal-B", "")
	if a == b || len(a) != 64 {
		t.Fatal("bad scope derivation")
	}
	if id.Digest(a, "pii:PERSON", "Alice") == id.Digest(a, "rule:keyword:PERSON", "Alice") {
		t.Fatal("canonical type collision")
	}
	for _, p := range []string{"principal-A", "Alice"} {
		raw, _ := os.ReadFile(ns.Registry.Path)
		if bytes.Contains(raw, []byte(p)) {
			t.Fatal("raw identity persisted")
		}
	}
	id.SessionScope = true
	if _, e = id.Scope("p", ""); e == nil {
		t.Fatal("empty root session accepted")
	}
}

// A Router that restarts waits for the one before it to let the lock go.
func TestFileWaitsForALockReleasedSoon(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry")
	f, e := OpenFile(path, "test", "key-id")
	if e != nil {
		t.Fatal(e)
	}
	go func() {
		time.Sleep(300 * time.Millisecond)
		_ = f.Close()
	}()
	reopened, e := OpenFile(path, "test", "key-id")
	if e != nil {
		t.Fatalf("reopen after release: %v", e)
	}
	_ = reopened.Close()
}
