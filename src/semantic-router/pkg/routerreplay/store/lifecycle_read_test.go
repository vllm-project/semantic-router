package store

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"io"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
)

func TestRedisReadsDrainBeforeClose(t *testing.T) {
	for _, method := range []string{"get", "list"} {
		t.Run(method, func(t *testing.T) {
			started := make(chan struct{})
			release := make(chan struct{})
			client := redis.NewClient(&redis.Options{Addr: "unused.invalid:6379"})
			client.AddHook(&blockingRedisReadHook{started: started, release: release})
			store := &RedisStore{client: client, lifecycle: newStorageLifecycle()}
			read := func() error {
				if method == "get" {
					_, _, err := store.Get(context.Background(), "replay-1")
					return err
				}
				_, err := store.List(context.Background())
				return err
			}
			assertReadDrainsBeforeClose(t, started, release, read, store.Close)
		})
	}
}

func TestPostgresReadsDrainBeforeClose(t *testing.T) {
	for _, method := range []string{"get", "list"} {
		t.Run(method, func(t *testing.T) {
			started := make(chan struct{})
			release := make(chan struct{})
			db := sql.OpenDB(&blockingSQLConnector{started: started, release: release})
			store := &PostgresStore{
				db:        db,
				tableName: "router_replay_records",
				lifecycle: newStorageLifecycle(),
			}
			read := func() error {
				if method == "get" {
					_, _, err := store.Get(context.Background(), "replay-1")
					return err
				}
				_, err := store.List(context.Background())
				return err
			}
			assertReadDrainsBeforeClose(t, started, release, read, store.Close)
		})
	}
}

func assertReadDrainsBeforeClose(
	t *testing.T,
	started <-chan struct{},
	release chan struct{},
	read func() error,
	closeStore func() error,
) {
	t.Helper()
	var releaseOnce sync.Once
	releaseRead := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseRead()

	readDone := make(chan error, 1)
	go func() { readDone <- read() }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("backend read did not start")
	}

	closeDone := make(chan error, 1)
	go func() { closeDone <- closeStore() }()
	select {
	case err := <-closeDone:
		t.Fatalf("Close() returned before the accepted read drained: %v", err)
	case <-time.After(20 * time.Millisecond):
	}

	releaseRead()
	select {
	case err := <-readDone:
		if errors.Is(err, ErrStorageClosed) {
			t.Fatalf("accepted read was rejected during Close(): %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("accepted read did not drain")
	}
	select {
	case err := <-closeDone:
		if err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close() did not finish after the read drained")
	}
	if err := read(); !errors.Is(err, ErrStorageClosed) {
		t.Fatalf("read after Close() error = %v, want ErrStorageClosed", err)
	}
}

type blockingRedisReadHook struct {
	once    sync.Once
	started chan struct{}
	release <-chan struct{}
}

func (h *blockingRedisReadHook) DialHook(next redis.DialHook) redis.DialHook {
	return func(ctx context.Context, network, address string) (net.Conn, error) {
		return next(ctx, network, address)
	}
}

func (h *blockingRedisReadHook) ProcessHook(_ redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, _ redis.Cmder) error {
		h.once.Do(func() { close(h.started) })
		select {
		case <-h.release:
			return redis.Nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func (h *blockingRedisReadHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

type blockingSQLConnector struct {
	once    sync.Once
	started chan struct{}
	release <-chan struct{}
}

func (c *blockingSQLConnector) Connect(context.Context) (driver.Conn, error) {
	return &blockingSQLConn{connector: c}, nil
}

func (c *blockingSQLConnector) Driver() driver.Driver {
	return blockingSQLDriver{}
}

type blockingSQLDriver struct{}

func (blockingSQLDriver) Open(string) (driver.Conn, error) {
	return nil, errors.New("blocking SQL driver must be opened through its connector")
}

type blockingSQLConn struct {
	connector *blockingSQLConnector
}

func (c *blockingSQLConn) Prepare(string) (driver.Stmt, error) {
	return nil, driver.ErrSkip
}

func (c *blockingSQLConn) Close() error {
	return nil
}

func (c *blockingSQLConn) Begin() (driver.Tx, error) {
	return nil, driver.ErrSkip
}

func (c *blockingSQLConn) QueryContext(
	ctx context.Context,
	_ string,
	_ []driver.NamedValue,
) (driver.Rows, error) {
	c.connector.once.Do(func() { close(c.connector.started) })
	select {
	case <-c.connector.release:
		return emptySQLRows{}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

type emptySQLRows struct{}

func (emptySQLRows) Columns() []string {
	return []string{"id"}
}

func (emptySQLRows) Close() error {
	return nil
}

func (emptySQLRows) Next([]driver.Value) error {
	return io.EOF
}
