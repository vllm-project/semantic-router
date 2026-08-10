package auth

import (
	"errors"
	"fmt"
	"net/http"
)

// MaxPasswordBytes is the maximum password length bcrypt accepts.
//
// The limit is on bytes, not characters: a 30 character password made of
// 3 byte runes is 90 bytes and is rejected, so callers must not substitute a
// character count for this check.
const MaxPasswordBytes = 72

// ErrPasswordTooLong is returned when a password cannot be hashed because it
// exceeds bcrypt's input limit. It is a caller mistake rather than a server
// fault, so handlers report it as 400 instead of 500.
var ErrPasswordTooLong = fmt.Errorf("password must be at most %d bytes", MaxPasswordBytes)

// ValidatePassword reports whether a password can be hashed. It is enforced
// inside Service.HashPassword so that every call site is covered, including
// ones added later, rather than relying on each handler to remember.
func ValidatePassword(password string) error {
	// len on a string is already the byte count, which is what bcrypt measures.
	if len(password) > MaxPasswordBytes {
		return ErrPasswordTooLong
	}
	return nil
}

// writePasswordHashError maps a hashing failure onto a status code. A password
// the caller can fix is a 400 carrying the reason; anything else is a 500 with
// a generic message, so internal errors are not echoed back to the client.
func writePasswordHashError(w http.ResponseWriter, err error) {
	if errors.Is(err, ErrPasswordTooLong) {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	http.Error(w, "failed to set password", http.StatusInternalServerError)
}
