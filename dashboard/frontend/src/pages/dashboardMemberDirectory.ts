import type { DashboardMember } from './AccessControlViewTypes'

const DASHBOARD_MEMBER_PAGE_SIZE = 200

interface DashboardMemberPage {
  users: DashboardMember[]
  total: number
  page: number
  limit: number
}

type DashboardMemberPageLoader = (
  page: number,
  limit: number,
  signal?: AbortSignal,
) => Promise<DashboardMemberPage>

const responseError = async (response: Response) =>
  (await response.text()).trim() || 'Could not load Dashboard identities'

export const requestDashboardMemberPage: DashboardMemberPageLoader = async (
  page,
  limit,
  signal,
) => {
  const params = new URLSearchParams({ page: String(page), limit: String(limit) })
  const response = await fetch(`/api/admin/users?${params}`, { signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as DashboardMemberPage
}

/**
 * Read the complete Dashboard identity directory through bounded server pages.
 * Pages are requested sequentially so a large workspace cannot create an
 * unbounded request burst against the control plane.
 */
export async function loadAllDashboardMembers(
  loadPage: DashboardMemberPageLoader = requestDashboardMemberPage,
  signal?: AbortSignal,
): Promise<DashboardMember[]> {
  const members = new Map<string, DashboardMember>()
  let pageNumber = 1
  let total = Number.POSITIVE_INFINITY

  while (members.size < total) {
    const page = await loadPage(pageNumber, DASHBOARD_MEMBER_PAGE_SIZE, signal)
    page.users.forEach((member) => members.set(member.id, member))
    total = Math.max(0, page.total)
    if (page.users.length === 0 || pageNumber * page.limit >= total) break
    pageNumber += 1
  }

  return Array.from(members.values())
}
