package auth

import (
	"errors"
	"fmt"
	"net/http"
)

// MaxPasswordBytes is bcrypt's input limit. It counts bytes, not characters.
const MaxPasswordBytes = 72

// ErrPasswordTooLong is a caller mistake, so handlers report it as 400.
var ErrPasswordTooLong = fmt.Errorf("password must be at most %d bytes", MaxPasswordBytes)

// ValidatePassword is enforced inside Service.HashPassword so every call site
// is covered rather than each handler remembering.
func ValidatePassword(password string) error {
	if len(password) > MaxPasswordBytes {
		return ErrPasswordTooLong
	}
	return nil
}

// writePasswordHashError keeps internal failures out of the response body.
func writePasswordHashError(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrPasswordTooLong) {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	http.Error(w, "failed to set password", http.StatusInternalServerError)
}
