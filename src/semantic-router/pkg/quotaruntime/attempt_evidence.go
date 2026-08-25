package quotaruntime

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"time"
)

const (
	maximumDispatchAttempts = uint32(6)
	maximumEvidenceIDBytes  = 256
	maximumEvidenceRevision = uint64(1<<32 - 1)
	unfinishedAttemptCode   = "attempt_unfinished"
)

func (e *RedisEngine) BeginDispatch(
	ctx context.Context,
	request BeginDispatchRequest,
) (BeginDispatchResult, error) {
	if err := validateBeginDispatchRequest(request); err != nil {
		return BeginDispatchResult{}, err
	}
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := dispatchAttemptKeys(partition, request.AdmissionID)
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return BeginDispatchResult{}, err
	}
	value, err := beginDispatchScript.Run(ctx, e.client, keys,
		request.AdmissionDigest,
		request.DispatchID,
		keyComponent(request.DispatchID),
		request.DispatchType,
		strconv.FormatUint(uint64(request.Ordinal), 10),
		request.DispatchPlanDigest,
		request.ModelID,
		strconv.FormatInt(request.ModelRevision, 10),
		request.RequestDigest,
		strconv.FormatInt(request.Deadline.UnixMilli(), 10),
		strconv.FormatUint(uint64(request.MaxAttempts), 10),
	).Result()
	if err != nil {
		return BeginDispatchResult{}, mapScriptError(err)
	}
	fields, err := scriptStrings(value, 5)
	if err != nil {
		return BeginDispatchResult{}, err
	}
	if fields[0] != "dispatch_started" {
		return BeginDispatchResult{}, fmt.Errorf(
			"%w: unexpected begin-dispatch state %q",
			ErrRuntimeCorrupt,
			fields[0],
		)
	}
	serverTime, err := parseMilliseconds(fields[2])
	if err != nil {
		return BeginDispatchResult{}, err
	}
	startedAt, err := parseMilliseconds(fields[3])
	if err != nil {
		return BeginDispatchResult{}, err
	}
	deadline, err := parseMilliseconds(fields[4])
	if err != nil {
		return BeginDispatchResult{}, err
	}
	if !deadline.Equal(request.Deadline) || startedAt.After(deadline) {
		return BeginDispatchResult{}, fmt.Errorf("%w: dispatch timestamps differ", ErrRuntimeCorrupt)
	}
	return BeginDispatchResult{
		MutationResult: MutationResult{Idempotent: fields[1] == "1", ServerTime: serverTime},
		StartedAt:      startedAt,
		Deadline:       deadline,
	}, nil
}

func (e *RedisEngine) BeginAttempt(
	ctx context.Context,
	request BeginAttemptRequest,
) (BeginAttemptResult, error) {
	if err := validateBeginAttemptRequest(request); err != nil {
		return BeginAttemptResult{}, err
	}
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := dispatchAttemptKeys(partition, request.AdmissionID)
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return BeginAttemptResult{}, err
	}
	value, err := beginAttemptScript.Run(ctx, e.client, keys,
		request.AdmissionDigest,
		request.DispatchID,
		keyComponent(request.DispatchID),
		request.DispatchPlanDigest,
		request.ModelID,
		strconv.FormatInt(request.ModelRevision, 10),
		request.RequestDigest,
		request.AttemptID,
		strconv.FormatUint(uint64(request.AttemptNumber), 10),
		request.BackendID,
		request.ProviderID,
	).Result()
	if err != nil {
		return BeginAttemptResult{}, mapScriptError(err)
	}
	fields, err := scriptStrings(value, 4)
	if err != nil {
		return BeginAttemptResult{}, err
	}
	if fields[0] != "attempt_started" || fields[1] != "0" {
		return BeginAttemptResult{}, fmt.Errorf(
			"%w: unexpected begin-attempt state",
			ErrRuntimeCorrupt,
		)
	}
	serverTime, err := parseMilliseconds(fields[2])
	if err != nil {
		return BeginAttemptResult{}, err
	}
	startedAt, err := parseMilliseconds(fields[3])
	if err != nil {
		return BeginAttemptResult{}, err
	}
	return BeginAttemptResult{
		MutationResult: MutationResult{ServerTime: serverTime},
		StartedAt:      startedAt,
	}, nil
}

func (e *RedisEngine) FinishAttempt(
	ctx context.Context,
	request FinishAttemptRequest,
) (FinishAttemptResult, error) {
	// Finishing an already-started attempt remains valid after its dispatch
	// deadline. This is a recovery operation that converges authoritative
	// evidence; BeginAttempt is the only operation that can create more work
	// and continues to reject expired dispatches.
	if err := validateFinishAttemptRequest(request); err != nil {
		return FinishAttemptResult{}, err
	}
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := dispatchAttemptKeys(partition, request.AdmissionID)
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return FinishAttemptResult{}, err
	}
	value, err := finishAttemptScript.Run(ctx, e.client, keys,
		request.AdmissionDigest,
		request.DispatchID,
		keyComponent(request.DispatchID),
		request.DispatchPlanDigest,
		request.ModelID,
		strconv.FormatInt(request.ModelRevision, 10),
		request.RequestDigest,
		request.AttemptID,
		strconv.FormatUint(uint64(request.AttemptNumber), 10),
		request.BackendID,
		request.ProviderID,
		string(request.State),
		strconv.Itoa(request.StatusCode),
		request.ErrorCode,
	).Result()
	if err != nil {
		return FinishAttemptResult{}, mapScriptError(err)
	}
	fields, err := scriptStrings(value, 4)
	if err != nil {
		return FinishAttemptResult{}, err
	}
	if fields[0] != "attempt_finished" {
		return FinishAttemptResult{}, fmt.Errorf(
			"%w: unexpected finish-attempt state %q",
			ErrRuntimeCorrupt,
			fields[0],
		)
	}
	serverTime, err := parseMilliseconds(fields[2])
	if err != nil {
		return FinishAttemptResult{}, err
	}
	completedAt, err := parseMilliseconds(fields[3])
	if err != nil {
		return FinishAttemptResult{}, err
	}
	return FinishAttemptResult{
		MutationResult: MutationResult{Idempotent: fields[1] == "1", ServerTime: serverTime},
		CompletedAt:    completedAt,
	}, nil
}

func (e *RedisEngine) ReadAttemptEvidence(
	ctx context.Context,
	request ReadAttemptEvidenceRequest,
) (ReadAttemptEvidenceResult, error) {
	// Evidence remains readable after the dispatch deadline so a replica crash
	// can be surfaced as unknown and fenced during settlement. Reading never
	// authorizes another attempt.
	if err := validateAttemptEvidenceReference(request.AttemptEvidenceReference); err != nil {
		return ReadAttemptEvidenceResult{}, err
	}
	partition, _ := newPartitionKeysWithPrefix(e.keyPrefix, request.Partition)
	keys := dispatchAttemptKeys(partition, request.AdmissionID)
	if err := validateRuntimeKeys(keys, partition.tag, e.keyPrefix); err != nil {
		return ReadAttemptEvidenceResult{}, err
	}
	value, err := readAttemptEvidenceScript.Run(ctx, e.client, keys,
		request.AdmissionDigest,
		request.DispatchID,
		keyComponent(request.DispatchID),
		strconv.FormatUint(uint64(request.Ordinal), 10),
		request.DispatchPlanDigest,
		request.ModelID,
		strconv.FormatInt(request.ModelRevision, 10),
	).Result()
	if err != nil {
		return ReadAttemptEvidenceResult{}, mapScriptError(err)
	}
	return parseAttemptEvidenceResult(request, value)
}

func parseAttemptEvidenceResult(
	request ReadAttemptEvidenceRequest,
	value any,
) (ReadAttemptEvidenceResult, error) {
	values, ok := value.([]any)
	if !ok || len(values) < 4 {
		return ReadAttemptEvidenceResult{}, fmt.Errorf(
			"%w: attempt evidence has an invalid response shape",
			ErrRuntimeCorrupt,
		)
	}
	header, err := scriptStrings(values[:4], 4)
	if err != nil {
		return ReadAttemptEvidenceResult{}, err
	}
	if header[0] != "attempt_evidence" {
		return ReadAttemptEvidenceResult{}, fmt.Errorf(
			"%w: unexpected attempt evidence state %q",
			ErrRuntimeCorrupt,
			header[0],
		)
	}
	serverTime, err := parseMilliseconds(header[1])
	if err != nil {
		return ReadAttemptEvidenceResult{}, err
	}
	revision, err := strconv.ParseUint(header[2], 10, 64)
	if err != nil || revision > maximumEvidenceRevision {
		return ReadAttemptEvidenceResult{}, fmt.Errorf("%w: invalid attempt evidence revision", ErrRuntimeCorrupt)
	}
	if header[3] == "0" {
		if len(values) != 4 {
			return ReadAttemptEvidenceResult{}, fmt.Errorf("%w: missing attempt evidence has trailing fields", ErrRuntimeCorrupt)
		}
		return ReadAttemptEvidenceResult{Revision: revision, ServerTime: serverTime}, nil
	}
	if header[3] != "1" || len(values) < 14 {
		return ReadAttemptEvidenceResult{}, fmt.Errorf("%w: invalid attempt evidence presence", ErrRuntimeCorrupt)
	}
	return parsePresentAttemptEvidence(request, values, revision, serverTime)
}

func parsePresentAttemptEvidence(
	request ReadAttemptEvidenceRequest,
	values []any,
	revision uint64,
	serverTime time.Time,
) (ReadAttemptEvidenceResult, error) {
	base, err := scriptStrings(values[:14], 14)
	if err != nil {
		return ReadAttemptEvidenceResult{}, err
	}
	dispatchOrdinal, err := parseUint32(base[5], "dispatch ordinal")
	if err != nil || dispatchOrdinal != request.Ordinal {
		return ReadAttemptEvidenceResult{}, fmt.Errorf("%w: dispatch ordinal differs", ErrRuntimeCorrupt)
	}
	modelRevision, err := strconv.ParseInt(base[8], 10, 64)
	if err != nil || modelRevision <= 0 {
		return ReadAttemptEvidenceResult{}, fmt.Errorf("%w: invalid model revision", ErrRuntimeCorrupt)
	}
	startedAt, err := parseMilliseconds(base[10])
	if err != nil {
		return ReadAttemptEvidenceResult{}, err
	}
	deadline, err := parseMilliseconds(base[11])
	if err != nil {
		return ReadAttemptEvidenceResult{}, err
	}
	maxAttempts, err := parseUint32(base[12], "maximum attempts")
	if err != nil || maxAttempts < 1 || maxAttempts > maximumDispatchAttempts {
		return ReadAttemptEvidenceResult{}, fmt.Errorf("%w: invalid maximum attempts", ErrRuntimeCorrupt)
	}
	attemptCount, err := parseUint32(base[13], "attempt count")
	if err != nil || attemptCount > maxAttempts || len(values) != 14+int(attemptCount)*9 {
		return ReadAttemptEvidenceResult{}, fmt.Errorf("%w: invalid attempt evidence count", ErrRuntimeCorrupt)
	}
	if base[6] != request.DispatchPlanDigest || base[7] != request.ModelID ||
		modelRevision != request.ModelRevision || !isSHA256Base64URL(base[9]) ||
		startedAt.After(deadline) {
		return ReadAttemptEvidenceResult{}, fmt.Errorf("%w: dispatch evidence differs", ErrRuntimeCorrupt)
	}

	evidence := DispatchAttemptEvidence{
		DispatchID:         request.DispatchID,
		DispatchType:       base[4],
		Ordinal:            dispatchOrdinal,
		DispatchPlanDigest: base[6],
		ModelID:            base[7],
		ModelRevision:      modelRevision,
		RequestDigest:      base[9],
		StartedAt:          startedAt,
		Deadline:           deadline,
		MaxAttempts:        maxAttempts,
		Attempts:           make([]AttemptEvidence, 0, attemptCount),
	}
	if err := validateBeginDispatchRequest(BeginDispatchRequest{
		DispatchReference: DispatchReference{
			Partition: request.Partition, AdmissionID: request.AdmissionID,
			AdmissionDigest: request.AdmissionDigest, DispatchID: request.DispatchID,
			DispatchPlanDigest: request.DispatchPlanDigest, ModelID: request.ModelID,
			ModelRevision: request.ModelRevision, RequestDigest: evidence.RequestDigest,
		},
		DispatchType: evidence.DispatchType,
		Ordinal:      evidence.Ordinal,
		Deadline:     evidence.Deadline,
		MaxAttempts:  evidence.MaxAttempts,
	}); err != nil {
		return ReadAttemptEvidenceResult{}, fmt.Errorf(
			"%w: invalid stored dispatch evidence: %w",
			ErrRuntimeCorrupt,
			err,
		)
	}
	for index := uint32(0); index < attemptCount; index++ {
		offset := 14 + int(index)*9
		fields, parseErr := scriptStrings(values[offset:offset+9], 9)
		if parseErr != nil {
			return ReadAttemptEvidenceResult{}, parseErr
		}
		attempt, parseErr := parseAttemptEvidence(index+1, fields)
		if parseErr != nil {
			return ReadAttemptEvidenceResult{}, parseErr
		}
		evidence.Attempts = append(evidence.Attempts, attempt)
	}
	return ReadAttemptEvidenceResult{
		Present: true, Revision: revision, Evidence: evidence, ServerTime: serverTime,
	}, nil
}

func dispatchAttemptKeys(partition partitionKeys, admissionID string) []string {
	return []string{
		partition.pending(admissionID),
		partition.terminal(admissionID),
		partition.dispatches(admissionID),
		partition.attempts(admissionID),
	}
}

func validateDispatchReference(reference DispatchReference) error {
	if err := validateEnvelope(reference.Partition, reference.AdmissionID, reference.AdmissionDigest); err != nil {
		return err
	}
	if !isSHA256Hex(reference.AdmissionDigest) {
		return fmt.Errorf("%w: admission digest must be 32-byte lowercase hex", ErrInvalidRequest)
	}
	if err := validateBoundedOpaque("dispatch ID", reference.DispatchID, maximumEvidenceIDBytes); err != nil {
		return err
	}
	if !isSHA256Hex(reference.DispatchPlanDigest) {
		return fmt.Errorf("%w: dispatch plan digest must be 32-byte lowercase hex", ErrInvalidRequest)
	}
	if err := validateBoundedOpaque("model ID", reference.ModelID, maximumEvidenceIDBytes); err != nil {
		return err
	}
	if reference.ModelRevision <= 0 {
		return fmt.Errorf("%w: model revision must be positive", ErrInvalidRequest)
	}
	if !isSHA256Base64URL(reference.RequestDigest) {
		return fmt.Errorf("%w: request digest must be canonical 32-byte base64url", ErrInvalidRequest)
	}
	return nil
}

func validateAttemptEvidenceReference(reference AttemptEvidenceReference) error {
	if err := validateEnvelope(reference.Partition, reference.AdmissionID, reference.AdmissionDigest); err != nil {
		return err
	}
	if !isSHA256Hex(reference.AdmissionDigest) {
		return fmt.Errorf("%w: admission digest must be 32-byte lowercase hex", ErrInvalidRequest)
	}
	if err := validateBoundedOpaque("dispatch ID", reference.DispatchID, maximumEvidenceIDBytes); err != nil {
		return err
	}
	if !isSHA256Hex(reference.DispatchPlanDigest) {
		return fmt.Errorf("%w: dispatch plan digest must be 32-byte lowercase hex", ErrInvalidRequest)
	}
	if err := validateBoundedOpaque("model ID", reference.ModelID, maximumEvidenceIDBytes); err != nil {
		return err
	}
	if reference.ModelRevision <= 0 {
		return fmt.Errorf("%w: model revision must be positive", ErrInvalidRequest)
	}
	return nil
}

func validateBeginDispatchRequest(request BeginDispatchRequest) error {
	if err := validateDispatchReference(request.DispatchReference); err != nil {
		return err
	}
	if err := validateBoundedOpaque("dispatch type", request.DispatchType, 64); err != nil {
		return err
	}
	if request.Deadline.IsZero() || request.Deadline.UnixMilli() <= 0 ||
		request.Deadline.Nanosecond()%int(time.Millisecond) != 0 {
		return fmt.Errorf("%w: deadline must be a positive millisecond-aligned instant", ErrInvalidRequest)
	}
	if request.MaxAttempts < 1 || request.MaxAttempts > maximumDispatchAttempts {
		return fmt.Errorf(
			"%w: maximum attempts must be between 1 and %d",
			ErrInvalidRequest,
			maximumDispatchAttempts,
		)
	}
	return nil
}

func validateBeginAttemptRequest(request BeginAttemptRequest) error {
	if err := validateDispatchReference(request.DispatchReference); err != nil {
		return err
	}
	return validateAttemptIdentity(
		request.AttemptID,
		request.AttemptNumber,
		request.BackendID,
		request.ProviderID,
	)
}

func validateFinishAttemptRequest(request FinishAttemptRequest) error {
	if err := validateDispatchReference(request.DispatchReference); err != nil {
		return err
	}
	if err := validateAttemptIdentity(
		request.AttemptID,
		request.AttemptNumber,
		request.BackendID,
		request.ProviderID,
	); err != nil {
		return err
	}
	if request.StatusCode != 0 && (request.StatusCode < 100 || request.StatusCode > 599) {
		return fmt.Errorf("%w: attempt status code is outside the HTTP range", ErrInvalidRequest)
	}
	switch request.State {
	case AttemptEvidenceKnownZero:
		if request.StatusCode != 0 {
			return fmt.Errorf("%w: known-zero attempt cannot carry a status code", ErrInvalidRequest)
		}
		if err := validateBoundedOpaque("attempt error code", request.ErrorCode, 128); err != nil {
			return err
		}
	case AttemptEvidenceResponseStarted:
		if request.StatusCode == 0 || request.ErrorCode != "" {
			return fmt.Errorf(
				"%w: response-started attempt requires status and no error code",
				ErrInvalidRequest,
			)
		}
	case AttemptEvidenceUnknown:
		if err := validateBoundedOpaque("attempt error code", request.ErrorCode, 128); err != nil {
			return err
		}
	default:
		return fmt.Errorf("%w: invalid attempt evidence state %q", ErrInvalidRequest, request.State)
	}
	return nil
}

func validateAttemptIdentity(attemptID string, number uint32, backendID, providerID string) error {
	if err := validateBoundedOpaque("attempt ID", attemptID, maximumEvidenceIDBytes); err != nil {
		return err
	}
	if number < 1 || number > maximumDispatchAttempts {
		return fmt.Errorf(
			"%w: attempt number must be between 1 and %d",
			ErrInvalidRequest,
			maximumDispatchAttempts,
		)
	}
	if err := validateBoundedOpaque("backend ID", backendID, maximumEvidenceIDBytes); err != nil {
		return err
	}
	return validateBoundedOpaque("provider ID", providerID, maximumEvidenceIDBytes)
}

func validateBoundedOpaque(label, value string, maximum int) error {
	if err := validateOpaque(label, value); err != nil {
		return err
	}
	if len(value) > maximum {
		return fmt.Errorf("%w: %s is too long", ErrInvalidRequest, label)
	}
	return nil
}

func isSHA256Hex(value string) bool {
	if len(value) != sha256.Size*2 || value != strings.ToLower(value) {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}

func isSHA256Base64URL(value string) bool {
	decoded, err := base64.RawURLEncoding.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size &&
		base64.RawURLEncoding.EncodeToString(decoded) == value
}

func parseUint32(value, label string) (uint32, error) {
	parsed, err := strconv.ParseUint(value, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("%w: invalid %s", ErrRuntimeCorrupt, label)
	}
	return uint32(parsed), nil
}

func parseAttemptEvidence(expectedNumber uint32, fields []string) (AttemptEvidence, error) {
	number, parseAttemptEvidenceErr := parseUint32(fields[1], "attempt number")
	if parseAttemptEvidenceErr != nil || number != expectedNumber {
		return AttemptEvidence{}, fmt.Errorf("%w: attempt numbers are not contiguous", ErrRuntimeCorrupt)
	}
	statusCode, parseAttemptEvidenceErr := strconv.Atoi(fields[5])
	if parseAttemptEvidenceErr != nil || statusCode < 0 || statusCode > 599 ||
		(statusCode != 0 && statusCode < 100) {
		return AttemptEvidence{}, fmt.Errorf("%w: invalid attempt status code", ErrRuntimeCorrupt)
	}
	startedAt, parseAttemptEvidenceErr := parseMilliseconds(fields[7])
	if parseAttemptEvidenceErr != nil {
		return AttemptEvidence{}, parseAttemptEvidenceErr
	}
	attempt := AttemptEvidence{
		AttemptID:     fields[0],
		AttemptNumber: number,
		BackendID:     fields[2],
		ProviderID:    fields[3],
		StatusCode:    statusCode,
		ErrorCode:     fields[6],
		StartedAt:     startedAt,
	}
	if err := validateAttemptIdentity(
		attempt.AttemptID,
		attempt.AttemptNumber,
		attempt.BackendID,
		attempt.ProviderID,
	); err != nil {
		return AttemptEvidence{}, fmt.Errorf("%w: invalid stored attempt identity: %w", ErrRuntimeCorrupt, err)
	}
	switch AttemptEvidenceState(fields[4]) {
	case AttemptEvidenceKnownZero:
		if attempt.StatusCode != 0 || strings.TrimSpace(attempt.ErrorCode) == "" {
			return AttemptEvidence{}, fmt.Errorf("%w: invalid known-zero attempt evidence", ErrRuntimeCorrupt)
		}
		attempt.State = AttemptEvidenceKnownZero
		attempt.Finished = true
	case AttemptEvidenceResponseStarted:
		if attempt.StatusCode == 0 || attempt.ErrorCode != "" {
			return AttemptEvidence{}, fmt.Errorf("%w: invalid response-started attempt evidence", ErrRuntimeCorrupt)
		}
		attempt.State = AttemptEvidenceResponseStarted
		attempt.Finished = true
	case AttemptEvidenceUnknown:
		if strings.TrimSpace(attempt.ErrorCode) == "" {
			return AttemptEvidence{}, fmt.Errorf("%w: invalid unknown attempt evidence", ErrRuntimeCorrupt)
		}
		attempt.State = AttemptEvidenceState(fields[4])
		attempt.Finished = true
	}
	if attempt.Finished {
		attempt.CompletedAt, parseAttemptEvidenceErr = parseMilliseconds(fields[8])
		if parseAttemptEvidenceErr != nil {
			return AttemptEvidence{}, parseAttemptEvidenceErr
		}
		if attempt.CompletedAt.Before(attempt.StartedAt) {
			return AttemptEvidence{}, fmt.Errorf("%w: attempt timestamps are reversed", ErrRuntimeCorrupt)
		}
	} else if AttemptEvidenceState(fields[4]) == "started" {
		if fields[8] != "" {
			return AttemptEvidence{}, fmt.Errorf("%w: unfinished attempt has completion time", ErrRuntimeCorrupt)
		}
		if attempt.StatusCode != 0 || attempt.ErrorCode != "" {
			return AttemptEvidence{}, fmt.Errorf("%w: unfinished attempt has terminal evidence", ErrRuntimeCorrupt)
		}
		attempt.State = AttemptEvidenceUnknown
		attempt.ErrorCode = unfinishedAttemptCode
		attempt.StatusCode = 0
	} else {
		return AttemptEvidence{}, fmt.Errorf("%w: invalid attempt evidence state", ErrRuntimeCorrupt)
	}
	return attempt, nil
}
