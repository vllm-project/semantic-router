package auth

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

func TestAdminMutationRevalidatesActorInsideWriteTransaction(t *testing.T) {
	for _, test := range []struct {
		operation string
		revoke    string
	}{
		{"role", "permission"},
		{"role", "session"},
		{"delete", "permission"},
		{"delete", "session"},
		{"password", "permission"},
		{"password", "session"},
	} {
		t.Run(test.operation+"/"+test.revoke, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "auth.db")
			store, err := NewStore(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = store.Close() })
			revoker, err := NewStore(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = revoker.Close() })
			svc := NewService(store, "admin-mutation-transaction-secret", 1)
			ctx := context.Background()
			if bootstrapErr := svc.EnsureBootstrapAdmin(ctx, "admin@example.test", "test-password", "Admin"); bootstrapErr != nil {
				t.Fatal(bootstrapErr)
			}
			token, admin, err := svc.Login(ctx, "admin@example.test", "test-password")
			if err != nil {
				t.Fatal(err)
			}
			claims, err := svc.ParseToken(token)
			if err != nil {
				t.Fatal(err)
			}
			target, err := store.CreateUser(ctx, "target@example.test", "Target", "old-hash", RoleRead, "active")
			if err != nil {
				t.Fatal(err)
			}
			actor := AuthContext{UserID: admin.ID, SessionID: claims.ID}

			// Keep the first store's only connection occupied until an independent
			// connection has committed the revocation. The admitted mutation then
			// proceeds and must observe it in its own write transaction.
			hold, err := store.db.BeginTx(ctx, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = hold.Rollback() }()
			waitBefore := store.db.Stats().WaitCount
			finished := make(chan error, 1)
			go func() {
				switch test.operation {
				case "role":
					_, mutationErr := store.UpdateUserRoleOrStatusAuthorized(ctx, actor, target.ID, RoleWrite, "active")
					finished <- mutationErr
				case "delete":
					finished <- store.DeleteUserAuthorized(ctx, actor, target.ID)
				case "password":
					finished <- store.UpdatePasswordAuthorized(ctx, actor, target.ID, "new-hash")
				}
			}()
			deadline := time.Now().Add(5 * time.Second)
			for store.db.Stats().WaitCount == waitBefore && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			if store.db.Stats().WaitCount == waitBefore {
				t.Fatal("admin mutation did not wait for the held connection")
			}
			if test.revoke == "session" {
				err = revoker.RevokeSession(ctx, claims.ID)
			} else {
				_, err = revoker.UpdateUserRoleOrStatus(ctx, admin.ID, RoleRead, "")
			}
			if err != nil {
				t.Fatal(err)
			}
			if rollbackErr := hold.Rollback(); rollbackErr != nil {
				t.Fatal(rollbackErr)
			}
			select {
			case mutationErr := <-finished:
				want := ErrPermissionDenied
				if test.revoke == "session" {
					want = errAdminSessionInvalid
				}
				if !errors.Is(mutationErr, want) {
					t.Fatalf("mutation after revocation = %v, want %v", mutationErr, want)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("admin mutation did not finish")
			}
			stored, err := store.GetUserByID(ctx, target.ID)
			if err != nil || stored.Role != RoleRead || stored.Status != "active" {
				t.Fatalf("revoked account mutation changed target: %+v, %v", stored, err)
			}
			var hash string
			if err := store.db.QueryRowContext(ctx, `SELECT password_hash FROM users WHERE id = ?`, target.ID).Scan(&hash); err != nil || hash != "old-hash" {
				t.Fatalf("revoked password mutation changed hash: %q, %v", hash, err)
			}
		})
	}
}

func TestAdminInvitationMutationRevalidatesActorInsideWriteTransaction(t *testing.T) {
	for _, test := range []struct {
		operation string
		revoke    string
	}{
		{"create", "permission"},
		{"create", "session"},
		{"rotate", "permission"},
		{"rotate", "session"},
		{"revoke", "permission"},
		{"revoke", "session"},
	} {
		t.Run(test.operation+"/"+test.revoke, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "auth.db")
			store, err := NewStore(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = store.Close() })
			revoker, err := NewStore(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = revoker.Close() })
			svc := NewService(store, "admin-invitation-transaction-secret", 1)
			ctx := context.Background()
			if bootstrapErr := svc.EnsureBootstrapAdmin(ctx, "admin@example.test", "test-password", "Admin"); bootstrapErr != nil {
				t.Fatal(bootstrapErr)
			}
			token, admin, err := svc.Login(ctx, "admin@example.test", "test-password")
			if err != nil {
				t.Fatal(err)
			}
			claims, err := svc.ParseToken(token)
			if err != nil {
				t.Fatal(err)
			}
			actor := AuthContext{UserID: admin.ID, SessionID: claims.ID}
			var invitation *Invitation
			var originalDigest string
			if test.operation != "create" {
				invitation, err = store.CreateInvitation(ctx, InvitationPersonal, "invitee@example.test", "Invitee", RoleAdmin, "original-digest", admin.ID, 1, time.Now().Add(time.Hour).Unix())
				if err != nil {
					t.Fatal(err)
				}
				_, originalDigest, err = store.GetInvitationByID(ctx, invitation.ID)
				if err != nil {
					t.Fatal(err)
				}
			}
			hold, err := store.db.BeginTx(ctx, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = hold.Rollback() }()
			waitBefore := store.db.Stats().WaitCount
			finished := make(chan error, 1)
			go func() {
				switch test.operation {
				case "create":
					_, mutationErr := store.CreateInvitation(ctx, InvitationPersonal, "next@example.test", "Next", RoleAdmin, "next-digest", admin.ID, 1, time.Now().Add(time.Hour).Unix(), actor)
					finished <- mutationErr
				case "rotate":
					_, mutationErr := store.RotateInvitation(ctx, invitation.ID, "rotated-digest", time.Now().Add(time.Hour).Unix(), actor)
					finished <- mutationErr
				case "revoke":
					finished <- store.RevokeInvitation(ctx, invitation.ID, actor)
				}
			}()
			deadline := time.Now().Add(5 * time.Second)
			for store.db.Stats().WaitCount == waitBefore && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			if store.db.Stats().WaitCount == waitBefore {
				t.Fatal("invitation mutation did not wait for the held connection")
			}
			if test.revoke == "session" {
				err = revoker.RevokeSession(ctx, claims.ID)
			} else {
				_, err = revoker.UpdateUserRoleOrStatus(ctx, admin.ID, RoleRead, "")
			}
			if err != nil {
				t.Fatal(err)
			}
			if rollbackErr := hold.Rollback(); rollbackErr != nil {
				t.Fatal(rollbackErr)
			}
			select {
			case mutationErr := <-finished:
				want := ErrPermissionDenied
				if test.revoke == "session" {
					want = errAdminSessionInvalid
				}
				if !errors.Is(mutationErr, want) {
					t.Fatalf("invitation mutation after revocation = %v, want %v", mutationErr, want)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("invitation mutation did not finish")
			}
			items, err := store.ListInvitations(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if test.operation == "create" {
				if len(items) != 0 {
					t.Fatalf("revoked actor created %d invitations", len(items))
				}
				return
			}
			if len(items) != 1 || items[0].Status != InvitationPending {
				t.Fatalf("revoked actor changed invitation: %+v", items)
			}
			_, digest, err := store.GetInvitationByID(ctx, invitation.ID)
			if err != nil || digest != originalDigest {
				t.Fatalf("revoked actor rotated invitation digest: %q, %v", digest, err)
			}
		})
	}
}
