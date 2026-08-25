package managementapi

import "time"

type UserCreateRequest struct {
	Email       string `json:"email"`
	DisplayName string `json:"displayName"`
}

type UserPatchRequest struct {
	Email       *string `json:"email,omitempty"`
	DisplayName *string `json:"displayName,omitempty"`
	Status      *string `json:"status,omitempty"`
}

type UserView struct {
	UserID      string     `json:"userId"`
	Email       string     `json:"email"`
	DisplayName string     `json:"displayName"`
	Status      string     `json:"status"`
	Revision    uint64     `json:"revision"`
	CreatedAt   time.Time  `json:"createdAt"`
	UpdatedAt   time.Time  `json:"updatedAt"`
	DeletedAt   *time.Time `json:"deletedAt,omitempty"`
}

type UserDetail struct {
	Data UserView `json:"data"`
}

type UserPage = Page[UserView]

type TeamCreateRequest struct {
	Name              string   `json:"name"`
	Description       string   `json:"description,omitempty"`
	AccessPolicyIDs   []string `json:"accessPolicyIds,omitempty"`
	RateLimitPolicyID *string  `json:"rateLimitPolicyId,omitempty"`
}

type TeamPatchRequest struct {
	Name        *string `json:"name,omitempty"`
	Description *string `json:"description,omitempty"`
	Status      *string `json:"status,omitempty"`
}

type TeamView struct {
	TeamID      string     `json:"teamId"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Status      string     `json:"status"`
	Revision    uint64     `json:"revision"`
	CreatedAt   time.Time  `json:"createdAt"`
	UpdatedAt   time.Time  `json:"updatedAt"`
	DeletedAt   *time.Time `json:"deletedAt,omitempty"`
}

type TeamDetail struct {
	Data TeamView `json:"data"`
}

type TeamPage = Page[TeamView]

type MembershipPutRequest struct {
	Role string `json:"role"`
}

type MembershipPatchRequest struct {
	Role   *string `json:"role,omitempty"`
	Status *string `json:"status,omitempty"`
}

type MembershipView struct {
	TeamID    string    `json:"teamId"`
	UserID    string    `json:"userId"`
	Role      string    `json:"role"`
	Status    string    `json:"status"`
	Revision  uint64    `json:"revision"`
	CreatedAt time.Time `json:"createdAt"`
	UpdatedAt time.Time `json:"updatedAt"`
}

type UserMembershipView struct {
	MembershipView
	TeamName   string `json:"teamName"`
	TeamStatus string `json:"teamStatus"`
}

type TeamMemberView struct {
	MembershipView
	DisplayName string `json:"displayName"`
	Email       string `json:"email"`
	UserStatus  string `json:"userStatus"`
}

type (
	UserMembershipPage = Page[UserMembershipView]
	TeamMemberPage     = Page[TeamMemberView]
)
