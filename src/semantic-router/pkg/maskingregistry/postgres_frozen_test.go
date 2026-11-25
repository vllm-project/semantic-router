package maskingregistry

import (
	"database/sql"
	"encoding/binary"
	"io"
	"net"
	"sync"
	"testing"
	"time"
)

// frozenPostgres accepts connections, completes the startup with
// AuthenticationOk and ReadyForQuery, and then never answers, as a Postgres
// that stopped (SIGSTOP, a stuck host) does after the connection is up.
func frozenPostgres(t *testing.T) (host, port string) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var conns []net.Conn
	t.Cleanup(func() {
		_ = ln.Close()
		mu.Lock()
		defer mu.Unlock()
		for _, c := range conns {
			_ = c.Close()
		}
	})
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			mu.Lock()
			conns = append(conns, c)
			mu.Unlock()
			go func() {
				var size [4]byte
				if _, err := io.ReadFull(c, size[:]); err != nil {
					return
				}
				if _, err := io.CopyN(io.Discard, c, int64(binary.BigEndian.Uint32(size[:]))-4); err != nil {
					return
				}
				_, _ = c.Write([]byte{'R', 0, 0, 0, 8, 0, 0, 0, 0, 'Z', 0, 0, 0, 5, 'I'})
				_, _ = io.Copy(io.Discard, c)
			}()
		}
	}()
	host, port, _ = net.SplitHostPort(ln.Addr().String())
	return host, port
}

func TestPostgresRegistryFailsWhenTheServerStopsAnswering(t *testing.T) {
	saved := queryTimeout
	queryTimeout = 200 * time.Millisecond
	t.Cleanup(func() { queryTimeout = saved })
	host, port := frozenPostgres(t)
	db, err := sql.Open("postgres", "host="+host+" port="+port+" user=router dbname=masking sslmode=disable")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	p := &Postgres{db: db, namespace: "test"}
	for name, call := range map[string]func() error{
		"prepare":  func() error { _, e := p.Prepare("scope", "pii:EMAIL", []int64{1}); return e },
		"allocate": func() error { _, e := p.Allocate("scope", "pii:EMAIL", "digest"); return e },
		"claim":    func() error { return p.Claim("pii:EMAIL", "[EMAIL_{index}]") },
	} {
		done := make(chan error, 1)
		go func() { done <- call() }()
		select {
		case err := <-done:
			if err == nil {
				t.Fatalf("%s: a server that never answers gave no error", name)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("%s: still waiting for a frozen server after 5 s, past the 200 ms limit", name)
		}
	}
}
