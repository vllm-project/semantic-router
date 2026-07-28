export type WebsiteMegaNavKey = 'docs' | 'research' | 'community' | 'blog'

export interface WebsiteMegaNavLink {
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
    description: 'Published research papers and talks from the project.',
    landingTo: '/publications',
    activePrefixes: [
      '/publications',
      '/white-paper',
      '/vision-paper',
    ],
    sections: [
      {
        key: 'published',
        title: 'Published work',
        description: 'Read the project thesis and peer-facing results.',
        links: [
          {
            key: 'publications',
            label: 'Papers & Talks',
            description: 'Browse publications, talks, and technical artifacts.',
            to: '/publications',
          },
          {
            key: 'white-paper',
            label: 'White Paper',
            description: 'Study the system design and engineering rationale.',
            to: '/white-paper',
          },
          {
            key: 'vision-paper',
            label: 'Vision Paper',
            description: 'See the long-range direction for intelligent routing.',
            to: '/vision-paper',
          },
        ],
      },
    ],
  },
  {
    key: 'community',
    label: 'Community',
    to: '/community/team',
    activePrefixes: ['/community'],
  },
  {
    key: 'blog',
    label: 'Blog',
    description: 'Engineering blog posts, release notes, and field reports.',
    landingTo: '/blog',
    activePrefixes: ['/blog'],
    sections: [],
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
