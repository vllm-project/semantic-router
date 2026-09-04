package auth

import (
	"errors"
	"fmt"
	"net/http"
)

// MaxPasswordBytes is bcrypt's input limit. It counts bytes, not characters.
const MaxPasswordBytes = 72

const MinPasswordCharacters = 9

// ErrPasswordTooLong is a caller mistake, so handlers report it as 400.
var (
	ErrPasswordTooLong  = fmt.Errorf("password must be at most %d bytes", MaxPasswordBytes)
	ErrPasswordTooShort = fmt.Errorf("password must be at least %d characters", MinPasswordCharacters)
)

// ValidatePassword is enforced inside Service.HashPassword so every call site
// is covered rather than each handler remembering.
func ValidatePassword(password string) error {
	if len([]rune(password)) < MinPasswordCharacters {
		return ErrPasswordTooShort
	}
	if len(password) > MaxPasswordBytes {
		return ErrPasswordTooLong
	}
	return nil
}

// writePasswordHashError keeps internal failures out of the response body.
func writePasswordHashError(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrPasswordTooLong) || errors.Is(err, ErrPasswordTooShort) {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	http.Error(w, "failed to set password", http.StatusInternalServerError)
}
