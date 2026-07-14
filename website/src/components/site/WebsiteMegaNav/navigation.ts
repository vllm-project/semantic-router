export type WebsiteMegaNavKey = 'docs' | 'research' | 'community'

export interface WebsiteMegaNavLink {
  key: string
  label: string
  description: string
  to?: string
  href?: string
}

export interface WebsiteMegaNavSection {
  key: string
  title: string
  description: string
  links: WebsiteMegaNavLink[]
}

export interface WebsiteMegaNavGroup {
  key: WebsiteMegaNavKey
  label: string
  description: string
  landingTo: string
  activePrefixes: string[]
  sections: WebsiteMegaNavSection[]
}

export const WEBSITE_MEGA_NAV_GROUPS: WebsiteMegaNavGroup[] = [
  {
    key: 'docs',
    label: 'Docs',
    description: 'The essential paths for learning, building, and operating.',
    landingTo: '/docs/intro',
    activePrefixes: ['/docs'],
    sections: [
      {
        key: 'start',
        title: 'Start here',
        description: 'Understand the project and get the router running.',
        links: [
          {
            key: 'quick-start',
            label: 'Quick Start',
            description: 'Understand the project and run the first route.',
            to: '/docs/intro',
          },
          {
            key: 'installation',
            label: 'Installation',
            description: 'Choose the deployment path that fits your stack.',
            to: '/docs/installation/',
          },
        ],
      },
      {
        key: 'build',
        title: 'Build',
        description: 'Configure the router and follow practical tutorials.',
        links: [
          {
            key: 'configuration',
            label: 'Configuration',
            description: 'Shape providers, signals, decisions, and plugins.',
            to: '/docs/installation/configuration',
          },
          {
            key: 'tutorials',
            label: 'Tutorials',
            description: 'Build routing behavior with guided examples.',
            to: '/docs/tutorials/algorithm/overview',
          },
        ],
      },
      {
        key: 'reference',
        title: 'Reference',
        description: 'Find API details and resolve common issues.',
        links: [
          {
            key: 'api',
            label: 'API Reference',
            description: 'Integrate with router and control-plane APIs.',
            to: '/docs/api/router',
          },
          {
            key: 'troubleshooting',
            label: 'Troubleshooting',
            description: 'Diagnose common runtime and deployment failures.',
            to: '/docs/troubleshooting/common-errors',
          },
        ],
      },
    ],
  },
  {
    key: 'research',
    label: 'Research',
    description: 'The project’s published research and engineering updates.',
    landingTo: '/publications',
    activePrefixes: [
      '/publications',
      '/white-paper',
      '/vision-paper',
      '/blog',
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
      {
        key: 'updates',
        title: 'Project updates',
        description: 'Follow engineering progress and field reports.',
        links: [
          {
            key: 'blog',
            label: 'Engineering Blog',
            description: 'Read release notes, experiments, and field reports.',
            to: '/blog',
          },
        ],
      },
    ],
  },
  {
    key: 'community',
    label: 'Community',
    description: 'People, governance, and open-source collaboration.',
    landingTo: '/community/team',
    activePrefixes: ['/community'],
    sections: [
      {
        key: 'project',
        title: 'Project',
        description: 'Understand who guides the project and how work is organized.',
        links: [
          {
            key: 'team',
            label: 'Project Team',
            description: 'Meet the maintainers and active committers.',
            to: '/community/team',
          },
          {
            key: 'steering',
            label: 'Steering Committee',
            description: 'Meet the Industry and Academic tracks.',
            to: '/community/steering-committee',
          },
          {
            key: 'governance',
            label: 'Roles & Governance',
            description: 'Read role, promotion, duty, and emeritus policy.',
            to: '/community/governance',
          },
        ],
      },
      {
        key: 'participate',
        title: 'Participate',
        description: 'Move from reading the project to shaping it.',
        links: [
          {
            key: 'working-groups',
            label: 'Working Groups',
            description: 'Find the group closest to your area of interest.',
            to: '/community/work-groups',
          },
          {
            key: 'contributing',
            label: 'Contributing Guide',
            description: 'Set up the workflow and prepare a strong change.',
            to: '/community/contributing',
          },
          {
            key: 'conduct',
            label: 'Code of Conduct',
            description: 'Review the expectations for collaboration.',
            to: '/community/code-of-conduct',
          },
        ],
      },
      {
        key: 'ecosystem',
        title: 'Ecosystem',
        description: 'Follow the code, models, and conversations around the router.',
        links: [
          {
            key: 'leaderboard',
            label: 'Contributor Leaderboard',
            description: 'Recognize the people advancing the project.',
            to: '/community/contributors',
          },
          {
            key: 'github',
            label: 'GitHub Repository',
            description: 'Browse source, releases, and project activity.',
            href: 'https://github.com/vllm-project/semantic-router',
          },
          {
            key: 'models',
            label: 'Models on Hugging Face',
            description: 'Discover the model artifacts used by the router.',
            href: 'https://huggingface.co/LLM-Semantic-Router',
          },
        ],
      },
    ],
  },
]

export function normalizeWebsitePath(pathname: string): string {
  const withoutLocale = pathname.replace(/^\/zh-Hans(?=\/|$)/i, '')
  const normalized = withoutLocale.replace(/\/+$/, '')
  return normalized || '/'
}

export function isWebsiteMegaNavGroupActive(
  group: WebsiteMegaNavGroup,
  pathname: string,
): boolean {
  const normalized = normalizeWebsitePath(pathname)
  return group.activePrefixes.some(
    prefix => normalized === prefix || normalized.startsWith(`${prefix}/`),
  )
}
