package recipe

import (
	"errors"
	"fmt"
)

const (
	ErrorInvalidRequest          = "invalid_request"
	ErrorInvalidURL              = "invalid_url"
	ErrorInsecureURL             = "insecure_url"
	ErrorSourceForbidden         = "source_forbidden"
	ErrorDownloadFailed          = "download_failed"
	ErrorDownloadTooLarge        = "download_too_large"
	ErrorArchiveDigestMismatch   = "archive_digest_mismatch"
	ErrorInvalidArchive          = "invalid_archive"
	ErrorInvalidRecipe           = "invalid_recipe"
	ErrorPackageVersionConflict  = "package_version_conflict"
	ErrorPackageNotFound         = "package_not_found"
	ErrorWarningsNotAcknowledged = "warnings_not_acknowledged"
	ErrorEnvironmentBinding      = "environment_binding_required"
	ErrorActivationConflict      = "activation_inconsistent"
	ErrorActivationConfirmation  = "activation_confirmation_required"
	ErrorActivationIncompatible  = "activation_incompatible"
	ErrorActivationFailed        = "activation_failed"
	ErrorActivationRollback      = "activation_rollback_failed"
	ErrorManagedStorageIsolation = "managed_storage_isolation_required"
)

// PackageError carries a stable public code and message while retaining an
// internal cause for logs. Error() intentionally returns only the safe text.
type PackageError struct {
	Code       string
	Status     int
	Safe       string
	underlying error
}

func (e *PackageError) Error() string {
	if e == nil {
		return ""
	}
	return e.Safe
}

func (e *PackageError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.underlying
}

func NewPackageError(code string, status int, safe string, cause error) error {
	return &PackageError{Code: code, Status: status, Safe: safe, underlying: cause}
}

func AsPackageError(err error) (*PackageError, bool) {
	var packageErr *PackageError
	return packageErr, errors.As(err, &packageErr)
}

func wrapPackageError(code string, status int, safe string, cause error) error {
	if cause == nil {
		cause = fmt.Errorf("%s", code)
	}
	return NewPackageError(code, status, safe, cause)
}
