package accesscontrol

import (
	"testing"

	"github.com/jackc/pgx/v5/pgconn"
)

func TestPublicErrorMapsValidationErrorsToBadRequest(t *testing.T) {
	status, message := PublicError(validationError("choose a valid policy"))
	if status != 400 || message != "choose a valid policy" {
		t.Fatalf("PublicError() = %d, %q", status, message)
	}
}

func TestPublicErrorMapsConstraintConflicts(t *testing.T) {
	status, message := PublicError(&pgconn.PgError{Code: "23505"})
	if status != 409 || message != "a resource with these details already exists" {
		t.Fatalf("PublicError(unique) = %d, %q", status, message)
	}
}
