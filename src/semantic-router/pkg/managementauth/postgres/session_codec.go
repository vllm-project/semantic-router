package postgres

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

type rowScanner interface {
	Scan(...any) error
}

type rowQueryer interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

type humanAssurance struct {
	AAL string   `json:"aal"`
	AMR []string `json:"amr"`
}

type workloadAssurance struct {
	Class string `json:"class"`
}

func getWith(ctx context.Context, queryer rowQueryer, sessionID string) (managementauth.LiveSession, error) {
	return scanLiveSession(queryer.QueryRowContext(ctx, liveSessionQuery, sessionID))
}

func scanLiveSession(row rowScanner) (managementauth.LiveSession, error) {
	var (
		live                managementauth.LiveSession
		issuerSessionID     sql.NullString
		authSourceID        sql.NullString
		authSourceKind      string
		evidenceKind        string
		assurance           []byte
		sourceAssuredAt     sql.NullTime
		status              string
		revokedAt           sql.NullTime
		principalStatus     string
		authSourceStatus    sql.NullString
		authSourceNotBefore sql.NullTime
		authSourceExpiresAt sql.NullTime
		authSourceAssuredAt sql.NullTime
	)
	if err := row.Scan(
		&live.ID,
		&live.PrincipalID,
		&issuerSessionID,
		&live.TokenID,
		&live.Audience,
		&authSourceKind,
		&authSourceID,
		&evidenceKind,
		&assurance,
		&live.AuthenticatedAt,
		&sourceAssuredAt,
		&live.ExpiresAt,
		&status,
		&revokedAt,
		&live.CreatedAt,
		&principalStatus,
		&authSourceStatus,
		&authSourceNotBefore,
		&authSourceExpiresAt,
		&authSourceAssuredAt,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return managementauth.LiveSession{}, managementauth.ErrSessionNotFound
		}
		return managementauth.LiveSession{}, fmt.Errorf("scan management session: %w", err)
	}
	if issuerSessionID.Valid {
		live.IssuerSessionID = &issuerSessionID.String
	}
	if authSourceID.Valid {
		live.AuthSourceID = authSourceID.String
	}
	live.AuthSourceKind = managementauth.AuthSourceKind(authSourceKind)
	live.EvidenceKind = managementauth.EvidenceKind(evidenceKind)
	live.Status = managementauth.SessionStatus(status)
	if revokedAt.Valid {
		value := revokedAt.Time.UTC()
		live.RevokedAt = &value
	}
	live.PrincipalStatus = managementauth.ResourceStatus(principalStatus)
	if authSourceStatus.Valid {
		live.AuthSourceStatus = managementauth.ResourceStatus(authSourceStatus.String)
	}
	setOptionalTime(&live.AuthSourceNotBefore, authSourceNotBefore)
	setOptionalTime(&live.AuthSourceExpiresAt, authSourceExpiresAt)
	setOptionalTime(&live.AuthSourceAssuredAt, authSourceAssuredAt)
	live.AuthenticatedAt = live.AuthenticatedAt.UTC()
	live.ExpiresAt = live.ExpiresAt.UTC()
	live.CreatedAt = live.CreatedAt.UTC()
	if err := decodeAssurance(&live, assurance, sourceAssuredAt); err != nil {
		return managementauth.LiveSession{}, err
	}
	return live, nil
}

func encodeAssurance(session managementauth.Session) ([]byte, any, error) {
	if session.EvidenceKind == managementauth.EvidenceHuman {
		encoded, err := json.Marshal(humanAssurance{AAL: session.Human.AAL, AMR: session.Human.AMR})
		return encoded, nil, err
	}
	if session.EvidenceKind == managementauth.EvidenceWorkload {
		encoded, err := json.Marshal(workloadAssurance{Class: session.Workload.Class})
		return encoded, time.Unix(session.Workload.SourceAssuredAt, 0).UTC(), err
	}
	return nil, nil, errors.New("management session evidence kind is invalid")
}

func decodeAssurance(live *managementauth.LiveSession, encoded []byte, sourceAssuredAt sql.NullTime) error {
	if live.EvidenceKind == managementauth.EvidenceHuman {
		var assurance humanAssurance
		if err := decodeStrict(encoded, &assurance); err != nil {
			return fmt.Errorf("decode management session human assurance: %w", err)
		}
		if sourceAssuredAt.Valid {
			return errors.New("human management session cannot have source_assured_at")
		}
		live.Human = &managementauth.HumanEvidence{
			AuthenticationTime: live.AuthenticatedAt.Unix(),
			AAL:                assurance.AAL,
			AMR:                assurance.AMR,
		}
		return nil
	}
	if live.EvidenceKind == managementauth.EvidenceWorkload {
		var assurance workloadAssurance
		if err := decodeStrict(encoded, &assurance); err != nil {
			return fmt.Errorf("decode management session workload assurance: %w", err)
		}
		if !sourceAssuredAt.Valid {
			return errors.New("workload management session requires source_assured_at")
		}
		live.Workload = &managementauth.WorkloadEvidence{
			Class:           assurance.Class,
			SourceAssuredAt: sourceAssuredAt.Time.Unix(),
		}
		return nil
	}
	return errors.New("management session evidence kind is invalid")
}

func decodeStrict(encoded []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return errors.New("management session assurance contains trailing data")
	}
	return nil
}

func setOptionalTime(target **time.Time, value sql.NullTime) {
	if value.Valid {
		normalized := value.Time.UTC()
		*target = &normalized
	}
}

func cloneStringPointer(value *string) *string {
	if value == nil {
		return nil
	}
	clone := *value
	return &clone
}

func cloneEvidence(
	human *managementauth.HumanEvidence,
	workload *managementauth.WorkloadEvidence,
) (*managementauth.HumanEvidence, *managementauth.WorkloadEvidence) {
	var humanClone *managementauth.HumanEvidence
	if human != nil {
		humanClone = &managementauth.HumanEvidence{
			AuthenticationTime: human.AuthenticationTime,
			AAL:                human.AAL,
			AMR:                slices.Clone(human.AMR),
		}
	}
	var workloadClone *managementauth.WorkloadEvidence
	if workload != nil {
		workloadClone = &managementauth.WorkloadEvidence{
			Class:           workload.Class,
			SourceAssuredAt: workload.SourceAssuredAt,
		}
	}
	return humanClone, workloadClone
}
