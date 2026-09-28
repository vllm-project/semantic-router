package internalauth

import "testing"

func TestProcessTokenAuthentication(t *testing.T) {
	if Token() == "" {
		t.Fatal("process token must not be empty")
	}
	if !Authenticate(Token()) {
		t.Fatal("generated process token was not authenticated")
	}
	if Authenticate("") {
		t.Fatal("empty token was authenticated")
	}
	if Authenticate("not-the-process-token") {
		t.Fatal("incorrect token was authenticated")
	}
}
