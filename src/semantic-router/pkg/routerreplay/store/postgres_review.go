package store

import (
	"context"
	"database/sql"
	"fmt"
)

// postgresReviewColumnsTemplate adds approval-review columns to new and
// existing tables. The review JSON is the record of truth; status, outcome
// and risk are copies kept in plain indexed columns for list filters.
const postgresReviewColumnsTemplate = `
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS request_kind VARCHAR(64);
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS review JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS review_status VARCHAR(32);
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS review_outcome VARCHAR(16);
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS review_risk VARCHAR(16);
		CREATE INDEX IF NOT EXISTS idx_{{table}}_request_kind_timestamp ON {{table}} (request_kind, timestamp DESC);
		CREATE INDEX IF NOT EXISTS idx_{{table}}_review_status_outcome ON {{table}} (review_status, review_outcome, review_risk);
	`

// postgresReviewSummaryColumn is the review a list page returns: everything
// but the planned action's full arguments. Path deletion fails on a JSON
// scalar, so rows without a review object read as NULL.
const postgresReviewSummaryColumn = `CASE WHEN jsonb_typeof(review) = 'object' THEN review #- '{action,arguments}' END`

// postgresReviewMetadataColumn is the review an aggregate scan reads: the
// status, verdict enums and latency, without action or rationale text.
const postgresReviewMetadataColumn = `CASE WHEN jsonb_typeof(review) = 'object' THEN (review - 'action') #- '{assessment,rationale}' END`

type postgresReviewColumns struct {
	requestKind sql.NullString
	reviewJSON  []byte
	status      sql.NullString
	outcome     sql.NullString
	risk        sql.NullString
}

func (columns postgresReviewColumns) assign(record *Record) error {
	if columns.requestKind.Valid {
		record.RequestKind = columns.requestKind.String
	}
	if err := unmarshalReplayOptionalJSON(columns.reviewJSON, &record.Review); err != nil {
		return fmt.Errorf("failed to unmarshal review: %w", err)
	}
	return nil
}

func reviewStatus(review *Review) string {
	if review == nil {
		return ""
	}
	return review.Status
}

// UpdateReview replaces the approval review of a record.
func (p *PostgresStore) UpdateReview(ctx context.Context, id string, review Review) error {
	release, err := p.lifecycle.beginMutation()
	if err != nil {
		return err
	}
	defer release()
	encoded, err := marshalReplayOptionalJSON(&review)
	if err != nil {
		return fmt.Errorf("failed to marshal review: %w", err)
	}
	encoded = sanitizePostgresJSON(encoded)
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		UPDATE %s
		SET review = $2,
		    review_status = $3,
		    review_outcome = $4,
		    review_risk = $5
		WHERE id = $1
	`, p.tableName)
	fn := func() error {
		result, err := p.db.ExecContext(ctx, query, id, encoded,
			emptyStringSQL(review.Status), emptyStringSQL(review.Outcome()), emptyStringSQL(review.RiskLevel()))
		if err != nil {
			return fmt.Errorf("failed to update review: %w", err)
		}
		rows, err := result.RowsAffected()
		if err != nil {
			return err
		}
		if rows == 0 {
			return fmt.Errorf("record with ID %s not found", id)
		}
		return nil
	}
	if p.asyncWrites {
		return p.runAsyncAndWait(ctx, fn)
	}
	return fn()
}
