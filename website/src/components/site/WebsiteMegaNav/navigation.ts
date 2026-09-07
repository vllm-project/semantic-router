export interface WebsitePrimaryNavItem {
  key: string
  label: string
  to: string
  activePrefixes: string[]
}

export const WEBSITE_PRIMARY_NAV_ITEMS: WebsitePrimaryNavItem[] = [
  {
    key: 'docs',
    label: 'Docs',
    to: '/docs/intro',
    activePrefixes: ['/docs'],
  },
  {
    key: 'research',
    label: 'Research',
    to: '/publications',
    activePrefixes: ['/publications', '/white-paper', '/vision-paper'],
  },
  {
    key: 'blog',
    label: 'Blog',
    to: '/blog',
    activePrefixes: ['/blog'],
  },
  {
    key: 'community',
    label: 'Community',
    to: '/community/team',
    activePrefixes: ['/community'],
  },
]

export function normalizeWebsitePath(pathname: string): string {
  const withoutLocale = pathname.replace(/^\/zh-Hans(?=\/|$)/i, '')
  const normalized = withoutLocale.replace(/\/+$/, '')
  return normalized || '/'
}

export function isWebsitePrimaryNavItemActive(
  item: WebsitePrimaryNavItem,
  pathname: string,
): boolean {
  const normalized = normalizeWebsitePath(pathname)
  return item.activePrefixes.some(
    prefix => normalized === prefix || normalized.startsWith(`${prefix}/`),
  )
}
