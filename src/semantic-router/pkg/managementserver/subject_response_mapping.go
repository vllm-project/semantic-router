package managementserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

func newUserView(value subjectmanagement.User) managementapi.UserView {
	return managementapi.UserView{
		UserID: value.ID, Email: value.Email, DisplayName: value.DisplayName,
		Status: string(value.Status), Revision: value.Revision, CreatedAt: value.CreatedAt,
		UpdatedAt: value.UpdatedAt, DeletedAt: cloneResponseTime(value.DeletedAt),
	}
}

func newUserPage(value subjectmanagement.Page[subjectmanagement.User]) managementapi.UserPage {
	items := make([]managementapi.UserView, len(value.Items))
	for index := range value.Items {
		items[index] = newUserView(value.Items[index])
	}
	return managementapi.UserPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
		TotalCount: pageTotalCount(value.TotalCount),
	}}
}

func newTeamView(value subjectmanagement.Team) managementapi.TeamView {
	return managementapi.TeamView{
		TeamID: value.ID, Name: value.Name, Description: value.Description,
		Status: string(value.Status), Revision: value.Revision, CreatedAt: value.CreatedAt,
		UpdatedAt: value.UpdatedAt, DeletedAt: cloneResponseTime(value.DeletedAt),
	}
}

func newTeamPage(value subjectmanagement.Page[subjectmanagement.Team]) managementapi.TeamPage {
	items := make([]managementapi.TeamView, len(value.Items))
	for index := range value.Items {
		items[index] = newTeamView(value.Items[index])
	}
	return managementapi.TeamPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
		TotalCount: pageTotalCount(value.TotalCount),
	}}
}

func newMembershipView(value subjectmanagement.Membership) managementapi.MembershipView {
	return managementapi.MembershipView{
		TeamID: value.TeamID, UserID: value.UserID, Role: string(value.Role),
		Status: string(value.Status), Revision: value.Revision,
		CreatedAt: value.CreatedAt, UpdatedAt: value.UpdatedAt,
	}
}

func newUserMembershipPage(value subjectmanagement.Page[subjectmanagement.UserMembership]) managementapi.UserMembershipPage {
	items := make([]managementapi.UserMembershipView, len(value.Items))
	for index := range value.Items {
		item := value.Items[index]
		items[index] = managementapi.UserMembershipView{
			MembershipView: newMembershipView(item.Membership),
			TeamName:       item.TeamName, TeamStatus: string(item.TeamStatus),
		}
	}
	return managementapi.UserMembershipPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
		TotalCount: pageTotalCount(value.TotalCount),
	}}
}

func newTeamMemberPage(value subjectmanagement.Page[subjectmanagement.TeamMember]) managementapi.TeamMemberPage {
	items := make([]managementapi.TeamMemberView, len(value.Items))
	for index := range value.Items {
		item := value.Items[index]
		items[index] = managementapi.TeamMemberView{
			MembershipView: newMembershipView(item.Membership),
			DisplayName:    item.DisplayName, Email: item.Email, UserStatus: string(item.UserStatus),
		}
	}
	return managementapi.TeamMemberPage{Data: items, Page: managementapi.PageInfo{
		NextCursor: value.NextCursor, HasMore: value.HasMore, PageSize: value.PageSize,
		TotalCount: pageTotalCount(value.TotalCount),
	}}
}
