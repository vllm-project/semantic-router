package historylock

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestASecondWriterWaitsForTheFirst(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "history")
	unlock, err := Lock(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info, err := os.Stat(dir); err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("history directory = %v, %v; want it private to its owner", info, err)
	}
	acquired := make(chan func(), 1)
	go func() {
		second, err := Lock(dir)
		if err != nil {
			t.Error(err)
			close(acquired)
			return
		}
		acquired <- second
	}()
	select {
	case <-acquired:
		t.Fatal("a second writer took the lock while the first held it")
	case <-time.After(100 * time.Millisecond):
	}
	unlock()
	select {
	case second, ok := <-acquired:
		if ok {
			second()
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the second writer did not take the released lock")
	}
}
