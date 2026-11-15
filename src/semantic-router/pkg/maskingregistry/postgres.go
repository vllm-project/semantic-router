package maskingregistry

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/postgres"
)

// Postgres uses the primary connection for membership and allocation alike.
// The counter row serializes reservations and allocations for a scope/type.
type Postgres struct {
	db        *sql.DB
	namespace string
}

func openPostgres(cfg *postgres.Config, name, identity string) (Store, error) {
	rc, e := postgres.NewRuntimeConfig(cfg, "masking_identity")
	if e != nil {
		return nil, e
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	db, e := postgres.OpenDB(ctx, rc)
	if e != nil {
		return nil, errors.New("masking registry: postgres unavailable")
	}
	p := &Postgres{db: db, namespace: name}
	if e = p.initialize(ctx, identity); e != nil {
		_ = db.Close()
		return nil, e
	}
	return p, nil
}

func (p *Postgres) initialize(ctx context.Context, identity string) error {
	// Fixed dedicated tables; unrelated store table_name settings do not alter
	// them. Replicas start together and CREATE TABLE IF NOT EXISTS races, so a
	// transaction lock lets one of them create the schema at a time.
	tx, e := p.db.BeginTx(ctx, nil)
	if e != nil {
		return fmt.Errorf("masking registry: postgres schema unavailable: %w", e)
	}
	defer func() { _ = tx.Rollback() }()
	if _, e = tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock(hashtext('vsr_masking_schema'))`); e == nil {
		_, e = tx.ExecContext(ctx, `
 CREATE TABLE IF NOT EXISTS vsr_masking_namespaces (namespace TEXT PRIMARY KEY, identity TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS vsr_masking_templates (namespace TEXT NOT NULL REFERENCES vsr_masking_namespaces(namespace), template TEXT NOT NULL, canonical_type TEXT NOT NULL, PRIMARY KEY(namespace,template));
 CREATE TABLE IF NOT EXISTS vsr_masking_counters (namespace TEXT NOT NULL REFERENCES vsr_masking_namespaces(namespace), scope TEXT NOT NULL, canonical_type TEXT NOT NULL, next_index BIGINT NOT NULL CHECK(next_index>=0), exhausted BOOLEAN NOT NULL DEFAULT FALSE, PRIMARY KEY(namespace,scope,canonical_type));
 CREATE TABLE IF NOT EXISTS vsr_masking_indices (namespace TEXT NOT NULL REFERENCES vsr_masking_namespaces(namespace), scope TEXT NOT NULL, canonical_type TEXT NOT NULL, idx BIGINT NOT NULL CHECK(idx>=0), digest TEXT, PRIMARY KEY(namespace,scope,canonical_type,idx), UNIQUE(namespace,scope,canonical_type,digest));`)
	}
	if e == nil {
		e = tx.Commit()
	}
	if e != nil {
		return fmt.Errorf("masking registry: postgres schema unavailable: %w", e)
	}
	_, e = p.db.ExecContext(ctx, `INSERT INTO vsr_masking_namespaces(namespace,identity) VALUES($1,$2) ON CONFLICT DO NOTHING`, p.namespace, identity)
	if e != nil {
		return e
	}
	var got string
	e = p.db.QueryRowContext(ctx, `SELECT identity FROM vsr_masking_namespaces WHERE namespace=$1`, p.namespace).Scan(&got)
	if e != nil || got != identity {
		return errors.New("masking registry: postgres namespace/key mismatch")
	}
	return nil
}

func (p *Postgres) Claim(typ, template string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, e := p.db.ExecContext(ctx, `INSERT INTO vsr_masking_templates(namespace,template,canonical_type) VALUES($1,$2,$3) ON CONFLICT DO NOTHING`, p.namespace, template, typ)
	if e != nil {
		return e
	}
	var owner string
	e = p.db.QueryRowContext(ctx, `SELECT canonical_type FROM vsr_masking_templates WHERE namespace=$1 AND template=$2`, p.namespace, template).Scan(&owner)
	if e != nil {
		return e
	}
	if owner != typ {
		return errors.New("masking registry: template belongs to another canonical type")
	}
	return nil
}

func (p *Postgres) transaction(ctx context.Context, scope, typ string) (*sql.Tx, int64, bool, error) {
	tx, e := p.db.BeginTx(ctx, nil)
	if e != nil {
		return nil, 0, false, e
	}
	// An acknowledged commit must reach WAL, even if a server default disabled it.
	_, e = tx.ExecContext(ctx, `SET LOCAL synchronous_commit = on`)
	if e == nil {
		_, e = tx.ExecContext(ctx, `INSERT INTO vsr_masking_counters(namespace,scope,canonical_type,next_index) VALUES($1,$2,$3,0) ON CONFLICT DO NOTHING`, p.namespace, scope, typ)
	}
	var next int64
	var exhausted bool
	if e == nil {
		e = tx.QueryRowContext(ctx, `SELECT next_index,exhausted FROM vsr_masking_counters WHERE namespace=$1 AND scope=$2 AND canonical_type=$3 FOR UPDATE`, p.namespace, scope, typ).Scan(&next, &exhausted)
	}
	if e != nil {
		_ = tx.Rollback()
		return nil, 0, false, e
	}
	return tx, next, exhausted, nil
}

func (p *Postgres) Prepare(scope, typ string, indices []int64) (map[int64]bool, error) {
	for _, i := range indices {
		if i < 0 {
			return nil, errors.New("masking registry: negative index")
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	tx, _, _, e := p.transaction(ctx, scope, typ)
	if e != nil {
		return nil, e
	}
	defer func() { _ = tx.Rollback() }()
	// Read membership before reserving, so an index reserved now reads as
	// not issued.
	rows, e := tx.QueryContext(ctx, `SELECT idx, digest IS NOT NULL FROM vsr_masking_indices WHERE namespace=$1 AND scope=$2 AND canonical_type=$3 AND idx = ANY($4)`, p.namespace, scope, typ, pq.Array(indices))
	if e != nil {
		return nil, e
	}
	issued := make(map[int64]bool, len(indices))
	for rows.Next() {
		var index int64
		var isIssued bool
		if e = rows.Scan(&index, &isIssued); e != nil {
			rows.Close()
			return nil, e
		}
		issued[index] = isIssued
	}
	if e = errors.Join(rows.Err(), rows.Close()); e != nil {
		return nil, e
	}
	_, e = tx.ExecContext(ctx, `INSERT INTO vsr_masking_indices(namespace,scope,canonical_type,idx) SELECT $1,$2,$3,unnest($4::bigint[]) ON CONFLICT DO NOTHING`, p.namespace, scope, typ, pq.Array(indices))
	if e != nil {
		return nil, e
	}
	if e = tx.Commit(); e != nil {
		return nil, e
	}
	return issued, nil
}

func (p *Postgres) Allocate(scope, typ, digest string) (int64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	tx, next, exhausted, e := p.transaction(ctx, scope, typ)
	if e != nil {
		return 0, e
	}
	defer func() { _ = tx.Rollback() }()
	var existing int64
	e = tx.QueryRowContext(ctx, `SELECT idx FROM vsr_masking_indices WHERE namespace=$1 AND scope=$2 AND canonical_type=$3 AND digest=$4`, p.namespace, scope, typ, digest).Scan(&existing)
	if e == nil {
		return existing, tx.Commit()
	}
	if !errors.Is(e, sql.ErrNoRows) {
		return 0, e
	}
	if exhausted {
		return 0, errors.New("masking registry: index exhausted")
	}
	// The lowest free index at or above next: next itself, or the end of the
	// run of taken indices that starts there.
	var free sql.NullInt64
	e = tx.QueryRowContext(ctx, `SELECT CASE WHEN NOT EXISTS (SELECT 1 FROM vsr_masking_indices WHERE namespace=$1 AND scope=$2 AND canonical_type=$3 AND idx=$4) THEN $4
 ELSE (SELECT MIN(a.idx) + 1 FROM vsr_masking_indices a WHERE a.namespace=$1 AND a.scope=$2 AND a.canonical_type=$3 AND a.idx >= $4 AND a.idx < 9223372036854775807
  AND NOT EXISTS (SELECT 1 FROM vsr_masking_indices b WHERE b.namespace=$1 AND b.scope=$2 AND b.canonical_type=$3 AND b.idx = a.idx + 1)) END`, p.namespace, scope, typ, next).Scan(&free)
	if e != nil {
		return 0, e
	}
	if !free.Valid {
		return 0, errors.New("masking registry: index exhausted")
	}
	index := free.Int64
	_, e = tx.ExecContext(ctx, `INSERT INTO vsr_masking_indices(namespace,scope,canonical_type,idx,digest) VALUES($1,$2,$3,$4,$5)`, p.namespace, scope, typ, index, digest)
	if e != nil {
		return 0, e
	}
	following := index
	if index != math.MaxInt64 {
		following++
	}
	_, e = tx.ExecContext(ctx, `UPDATE vsr_masking_counters SET next_index=$4,exhausted=$5 WHERE namespace=$1 AND scope=$2 AND canonical_type=$3`, p.namespace, scope, typ, following, index == math.MaxInt64)
	if e != nil {
		return 0, e
	}
	// After an ambiguous commit the next attempt finds digest under the lock.
	if e = tx.Commit(); e != nil {
		return 0, e
	}
	return index, nil
}
