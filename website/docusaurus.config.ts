import type { Config } from '@docusaurus/types'
import type * as Preset from '@docusaurus/preset-classic'
import type { PrismTheme } from 'prism-react-renderer'
import { SITE_SOCIAL_PREVIEW_IMAGE } from './src/data/socialPreview'
import blogSearchIndexPlugin from './src/plugins/blogSearchIndex'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

// Code blocks are ink in both colour modes, in System Duo's syntax colours (each at least 5:1 on ink), the colours the
// homepage's terminals use.
const codeTheme: PrismTheme = {
  plain: { color: '#d9dbe0', backgroundColor: '#16181d' },
  styles: [
    { types: ['comment', 'prolog', 'doctype', 'cdata'], style: { color: '#8b8f99' } },
    { types: ['string', 'char', 'attr-value', 'regex', 'inserted', 'url'], style: { color: '#f3d3a8' } },
    { types: ['property', 'attr-name', 'tag', 'selector', 'key', 'atrule'], style: { color: '#9ec1ef' } },
    { types: ['number', 'boolean', 'constant', 'symbol'], style: { color: '#f6a37f' } },
    { types: ['keyword', 'important', 'builtin', 'deleted'], style: { color: '#ff8c69' } },
    { types: ['function', 'class-name'], style: { color: '#ffffff', fontWeight: '600' } },
    { types: ['operator', 'punctuation', 'entity'], style: { color: '#b8c6dc' } },
    { types: ['variable', 'parameter', 'environment'], style: { color: '#ffcf7a' } },
  ],
}
const siteUrl = 'https://vllm-sr.ai'
const siteDefaultDescription
  = 'An open, programmable decision layer for models and compute.'
const siteSocialTitle
  = 'Intelligence Beyond Any One Model | vLLM Semantic Router'
const siteSocialPreviewImageUrl = `${siteUrl}/${SITE_SOCIAL_PREVIEW_IMAGE}`

// Docs pages that moved, as paths under /docs. Until the current docs became
// the default version, they were served under /docs/next and the v0.4 docs
// under /docs, so each old path redirects from both.
const movedDocs: { from: string[], to: string }[] = [
  {
    from: [
      'tutorials/signal/learned/classifier',
      'tutorials/signal/learned/preference',
      'tutorials/signal/learned/kb',
      'tutorials/signal/learned/embedding',
      'tutorials/signal/learned/embedding-design-principles',
      'tutorials/signal/learned/decision',
      'tutorials/signal/learned/systemone',
      'tutorials/signal/heuristic/action',
    ],
    to: 'tutorials/signal/systemone',
  },
  {
    from: [
      'tutorials/signal/learned/reask',
      'tutorials/signal/heuristic/topic-continuity',
    ],
    to: 'tutorials/signal/application/turn-relation',
  },
  // The application signals moved from signal/learned to
  // signal/application.
  ...[
    'complexity',
    'domain',
    'fact-check',
    'hallucination',
    'jailbreak',
    'modality',
    'pii',
    'refusal',
    'safety',
    'turn-relation',
    'user-feedback',
  ].map(signal => ({
    from: [`tutorials/signal/learned/${signal}`],
    to: `tutorials/signal/application/${signal}`,
  })),
  {
    from: ['tutorials/signal/heuristic/modality'],
    to: 'tutorials/signal/application/modality',
  },
  // Pages folded into the System One chapter and the signal pages.
  {
    from: ['model-runtime/guides/decisions'],
    to: 'system-one/overview',
  },
  {
    from: ['model-runtime/guides/classify'],
    to: 'tutorials/signal/overview',
  },
  {
    from: ['model-runtime/guides/pii'],
    to: 'tutorials/signal/application/pii',
  },
  {
    from: ['model-runtime/guides/hallucination'],
    to: 'tutorials/signal/application/hallucination',
  },
  {
    from: ['tutorials/global/vela-models'],
    to: 'system-one/deploy/deployments',
  },
  // Removed algorithms: the migration guide lists each rewrite.
  {
    from: [
      'tutorials/algorithm/looper/confidence',
      'tutorials/algorithm/native/cascade',
    ],
    to: 'tutorials/algorithm/looper/escalate',
  },
  {
    from: [
      'tutorials/algorithm/selection/automix',
      'tutorials/algorithm/selection/hybrid',
    ],
    to: 'tutorials/algorithm/selection/multi-factor',
  },
  {
    from: [
      'tutorials/algorithm/selection/prompt',
      'tutorials/algorithm/selection/router-dc',
      'tutorials/algorithm/selection/knn',
      'tutorials/algorithm/selection/kmeans',
      'tutorials/algorithm/selection/svm',
      'tutorials/algorithm/selection/mlp',
      'tutorials/algorithm/selection/decision',
      'training/ml-model-selection',
    ],
    to: 'tutorials/algorithm/selection/systemone',
  },
  {
    from: [
      'training/classifier-models',
      'training/mmbert-safety-classifier',
      'training/multimodal-embeddings',
      'training/model-catalog',
    ],
    to: 'training/training-overview',
  },
  {
    from: [
      'installation/native-backends',
      'installation/runtime/engines-and-hardware',
    ],
    to: 'model-runtime/overview',
  },
  {
    from: [
      'installation/runtime/in-process',
      'installation/runtime/models-and-bindings',
    ],
    to: 'model-runtime/deploy/connect-the-router',
  },
  {
    from: ['installation/runtime/openvino'],
    to: 'model-runtime/deploy/migrate',
  },
  {
    from: [
      'installation/runtime/safety',
      'tutorials/global/safety-models-and-policy',
      'model-runtime/guides/safety',
    ],
    to: 'tutorials/signal/application/safety',
  },
  {
    from: [
      'tutorials/plugin/rag',
      'tutorials/plugin/memory',
      'tutorials/plugin/context-compression',
      'model-runtime/guides/rerank',
    ],
    to: 'tutorials/plugin/context',
  },
  {
    from: ['tutorials/plugin/shadow-dispatch'],
    to: 'tutorials/plugin/router-replay',
  },
  {
    from: ['tutorials/plugin/tool-selection'],
    to: 'tutorials/plugin/tools',
  },
  {
    from: [
      'tutorials/plugin/hallucination',
      'tutorials/plugin/response-jailbreak',
    ],
    to: 'tutorials/plugin/response-guard',
  },
  {
    from: ['tutorials/plugin/semantic-cache'],
    to: 'tutorials/plugin/response-cache',
  },
  {
    from: [
      'installation/runtime/embeddings',
      'tutorials/global/remote-embeddings',
    ],
    to: 'model-runtime/guides/embeddings',
  },
  {
    from: ['installation/runtime/lifecycle-diagnostics'],
    to: 'model-runtime/deploy/troubleshooting',
  },
  {
    from: ['model-runtime/profiles'],
    to: 'model-runtime/deploy/engine-args',
  },
  {
    from: [
      'tutorials/signal/heuristic/event',
      'tutorials/signal/heuristic/structure',
    ],
    to: 'tutorials/signal/heuristic/keyword',
  },
  {
    from: ['installation/kubernetes'],
    to: 'installation/k8s/ai-gateway',
  },
  {
    from: [
      'cli/overview',
      'cli/commands-reference',
    ],
    to: 'installation/',
  },
  {
    from: ['installation/models-and-recipes'],
    to: 'tutorials/global/models-entrypoints-serving',
  },
  {
    from: ['cli/troubleshooting'],
    to: 'troubleshooting/common-errors',
  },
  {
    from: ['tutorials/signal/projections'],
    to: 'tutorials/projection/overview',
  },
  {
    from: ['overview/collective-intelligence'],
    to: 'overview/mom-model-family',
  },
  {
    from: ['troubleshooting/vsr-headers'],
    to: 'api/headers',
  },
  // System One, Model Runtime and the global settings regrouped into folders.
  // The two section overviews keep their paths with a slug.
  {
    from: ['tutorials/global/overview'],
    to: 'installation/global-settings',
  },
  {
    from: ['system-one/concepts'],
    to: 'system-one/get-started/concepts',
  },
  {
    from: ['system-one/api'],
    to: 'system-one/use/api',
  },
  {
    from: ['system-one/router'],
    to: 'system-one/use/router',
  },
  {
    from: ['system-one/custom-questions'],
    to: 'system-one/use/custom-questions',
  },
  {
    from: ['system-one/deployments'],
    to: 'system-one/deploy/deployments',
  },
  {
    from: ['system-one/performance'],
    to: 'system-one/deploy/performance',
  },
  {
    from: ['system-one/reference'],
    to: 'system-one/reference/presets',
  },
  {
    from: ['model-runtime/quickstart'],
    to: 'model-runtime/get-started/quickstart',
  },
  {
    from: ['model-runtime/choose-a-model'],
    to: 'model-runtime/get-started/choose-a-model',
  },
  {
    from: [
      'model-runtime/deploy',
      'tutorials/global/model-runtime',
    ],
    to: 'model-runtime/deploy/connect-the-router',
  },
  {
    from: ['model-runtime/engine-args'],
    to: 'model-runtime/deploy/engine-args',
  },
  {
    from: ['installation/runtime/external'],
    to: 'model-runtime/deploy/remote-services',
  },
  {
    from: ['model-runtime/migrate'],
    to: 'model-runtime/deploy/migrate',
  },
  {
    from: ['model-runtime/troubleshooting'],
    to: 'model-runtime/deploy/troubleshooting',
  },
  {
    from: ['model-runtime/plugins'],
    to: 'model-runtime/extend/plugins',
  },
  {
    from: ['model-runtime/reference'],
    to: 'model-runtime/reference/http-api',
  },
]

// Pages the current docs dropped without a replacement. The v0.4 docs still
// have them, so their old paths under /docs lead there.
const releasedOnlyDocs = [
  'fleet-sim/getting-started',
  'fleet-sim/guide',
  'fleet-sim/overview',
  'fleet-sim/power-model',
  'fleet-sim/related-work',
  'fleet-sim/sim-algorithms',
  'fleet-sim/use-cases',
  'tutorials/algorithm/selection/gmtrouter',
]

const config: Config = {
  title: 'vLLM Semantic Router',
  tagline: 'An open, programmable decision layer for models and compute.',
  favicon: 'img/vllm.png',

  // Set the production url of your site here
  url: siteUrl,
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'vllm-project', // Usually your GitHub org/user name.
  projectName: 'semantic-router', // Usually your repo name.

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
    localeConfigs: {
      'en': { label: 'English', htmlLang: 'en-US' },
      'zh-Hans': {
        label: '简体中文',
        htmlLang: 'zh-Hans',
        baseUrl: '/zh-hans/',
      },
    },
  },

  markdown: {
    mermaid: true,
    format: 'mdx',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  themes: [
    '@docusaurus/theme-mermaid',
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true, // cache-bust the index between deploys
        indexDocs: true,
        indexBlog: true,
        indexPages: false, // homepage/community are marketing pages, not docs
        docsRouteBasePath: '/docs',
        blogRouteBasePath: '/blog',
        searchBarShortcut: true,
        searchBarShortcutHint: true,
        // The site ships a full zh-Hans locale, so the index needs a Chinese
        // tokenizer as well: Chinese is written without spaces and the default
        // English tokenizer cannot split it. "zh" pulls in @node-rs/jieba, a
        // native module that ships prebuilt binaries.
        language: ['en', 'zh'],
        // v1 scope: index the default docs version only (the latest docs, from
        // main), per the decision on #2737.
        // To make the released versions searchable later, drop this and add
        // searchContextByPaths: ['docs', 'docs/v0.4', 'docs/v0.3', 'docs/v0.2', 'docs/v0.1']
        // so results stay scoped to the version the reader is on.
        // NOTE: ignoreFiles matches the route *without* a leading slash (the
        // plugin strips baseUrl, which is "/" here, off the front) and without
        // the base URL itself, so the pattern must not anchor on "/".
        ignoreFiles: [/^docs\/v\d+\.\d+\//],
        // styling is handled in a later phase
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          beforeDefaultRemarkPlugins: [[remarkMath, {}]],
          beforeDefaultRehypePlugins: [[rehypeKatex, {}]],
          sidebarPath: './sidebars.ts',
          lastVersion: 'current',
          versions: {
            'current': {
              label: 'Latest (main)',
              path: '',
              banner: 'none',
              badge: true,
            },
            'v0.4': {
              label: 'v0.4',
              path: 'v0.4',
              badge: true,
            },
            'v0.3': {
              label: 'v0.3',
              path: 'v0.3',
              badge: true,
            },
            'v0.2': {
              label: 'v0.2',
              path: 'v0.2',
              badge: true,
            },
            'v0.1': {
              label: 'v0.1',
              path: 'v0.1',
              badge: true,
            },
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // Custom editUrl function to always point to the "current" (main) version
          editUrl: ({ locale, docPath }) => {
            if (locale !== 'en') {
              return `https://github.com/vllm-project/semantic-router/edit/main/website/i18n/${locale}/docusaurus-plugin-content-docs/current/${docPath}`
            }
            return `https://github.com/vllm-project/semantic-router/edit/main/website/docs/${docPath}`
          },
        },
        blog: {
          showReadingTime: true,
          postsPerPage: 10,
          blogTitle: 'vLLM Semantic Router Blog',
          blogDescription:
            'Latest updates, insights, and technical articles about vLLM Semantic Router',
          blogSidebarTitle: 'Recent Posts',
          blogSidebarCount: 10,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/vllm-project/semantic-router/tree/main/website/blog/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
          filename: 'sitemap.xml',
          ignorePatterns: ['/tags/**', '/search'],
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    // Publishes every published blog post as global data so the blog list page can
    // search the whole archive instead of just the posts on the current page.
    blogSearchIndexPlugin,
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          {
            from: '/blog/introduce-vela',
            to: '/blog/vela-models',
          },
          {
            from: '/blog/multi-objective-mom-on-amd-developer-cloud',
            to: '/docs/overview/mom-model-family',
          },
          ...movedDocs.map(({ from, to }) => ({
            from: from.flatMap(path => [`/docs/${path}`, `/docs/next/${path}`]),
            to: `/docs/${to}`,
          })),
          ...releasedOnlyDocs.map(path => ({
            from: `/docs/${path}`,
            to: `/docs/v0.4/${path}`,
          })),
        ],
        // Every current page was served under /docs/next until the current
        // docs became the default version.
        createRedirects(existingPath: string) {
          if (existingPath.startsWith('/docs/') && !/^\/docs\/v\d+\.\d+(\/|$)/.test(existingPath)) {
            return existingPath.replace('/docs/', '/docs/next/')
          }
          return undefined
        },
      },
    ],
  ],

  themeConfig: {
    image: SITE_SOCIAL_PREVIEW_IMAGE,
    metadata: [
      { name: 'description', content: siteDefaultDescription },
      {
        name: 'keywords',
        content:
          'programmable decision layer, agent harness, models and compute, Mixture-of-Models, open-source LLM router, multi-model routing, model selection, bounded model collaboration, semantic router, vLLM',
      },
      { name: 'author', content: 'vLLM Semantic Router Team' },
      { name: 'application-name', content: 'vLLM Semantic Router' },
      { property: 'og:title', content: siteSocialTitle },
      { property: 'og:description', content: siteDefaultDescription },
      { property: 'og:type', content: 'website' },
      { property: 'og:site_name', content: 'vLLM Semantic Router' },
      { property: 'og:image', content: siteSocialPreviewImageUrl },
      {
        property: 'og:image:alt',
        content: 'vLLM Semantic Router social preview',
      },
      { name: 'twitter:card', content: 'summary_large_image' },
      { name: 'twitter:title', content: siteSocialTitle },
      { name: 'twitter:description', content: siteDefaultDescription },
      { name: 'twitter:image', content: siteSocialPreviewImageUrl },
      {
        name: 'twitter:image:alt',
        content: 'vLLM Semantic Router social preview',
      },

      // GEO metadata config
      { name: 'geo.region', content: 'US-CA' },
      { name: 'geo.placename', content: 'San Francisco' },
      { name: 'geo.position', content: '37.7749;-122.4194' },
      { name: 'ICBM', content: '37.7749, -122.4194' },
    ],
    navbar: {
      // Navbar and logo follow the reader's color mode on every page.
      logo: {
        alt: 'vLLM Semantic Router Logo',
        src: 'img/vllm-sr-logo.light.png',
        srcDark: 'img/vllm-sr-logo.white.png',
      },
      items: [
        {
          type: 'localeDropdown',
          className: 'nav-locale',
          position: 'right',
        },
        {
          type: 'docsVersionDropdown',
          className: 'nav-docs-only',
          position: 'right',
          dropdownActiveClassDisabled: true,
        },
        {
          label: 'Docs',
          to: '/docs/intro',
          className: 'nav-primary',
          position: 'left',
        },
        {
          label: 'Research',
          to: '/publications',
          className: 'nav-primary',
          position: 'left',
        },
        {
          label: 'Blog',
          to: '/blog',
          className: 'nav-primary',
          position: 'left',
        },
        {
          label: 'Community',
          to: '/community/team',
          className: 'nav-primary',
          position: 'left',
        },
        // Secondary destinations live in one overflow menu so the bar fits on a
        // 1024px laptop instead of collapsing to a hamburger at 1340px.
        {
          type: 'dropdown',
          label: 'More',
          className: 'nav-utility',
          position: 'right',
          items: [
            {
              label: 'Model Hub',
              to: '/models',
            },
            {
              label: 'Leaderboard',
              to: '/community/contributors',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/vllm-project/semantic-router',
            },
            {
              label: 'Hugging Face',
              href: 'https://huggingface.co/vllm-sr',
            },
            {
              label: 'Discussions',
              href: 'https://github.com/vllm-project/semantic-router/discussions',
            },
            // The locale dropdown above gives up its place from 997px to
            // 1339px; shell.css shows these links only in that range.
            {
              type: 'custom-localeLinks',
              className: 'nav-locale-more',
            },
          ],
        },
        {
          label: 'System Duo',
          href: 'https://app.vllm-sr.ai/',
          className: 'nav-dashboard-cta',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'Quick Start',
              to: '/docs/installation',
            },
            {
              label: 'Governance',
              to: '/community/governance',
            },
            {
              label: 'Contributing',
              to: '/community/contributing',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/vllm-project/semantic-router',
            },
            {
              label: 'Hugging Face',
              href: 'https://huggingface.co/vllm-sr',
            },
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/vllm-project/semantic-router/discussions',
            },
            {
              label: 'Leaderboard',
              to: '/community/contributors',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'Publications',
              to: '/publications',
            },
            {
              label: 'White Paper',
              to: '/white-paper',
            },
            {
              label: 'Vision Paper',
              to: '/vision-paper',
            },
            {
              label: 'License',
              href: 'https://github.com/vllm-project/semantic-router/blob/main/LICENSE',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} vLLM Semantic Router Team.`,
    },
    prism: {
      theme: codeTheme,
      darkTheme: codeTheme,
      additionalLanguages: ['bash', 'json', 'yaml', 'go', 'rust', 'python'],
    },
    colorMode: {
      // Only applies when the OS states no preference.
      defaultMode: 'light',
      disableSwitch: false,
      // Sets the default from the OS. Upstream also uses this flag to make the
      // navbar button a three-way cycle; the swizzle in
      // `src/theme/Navbar/ColorModeToggle` keeps it at two states.
      respectPrefersColorScheme: true,
    },
    mermaid: {
      theme: {
        light: 'neutral',
        dark: 'dark',
      },
    },
  } satisfies Preset.ThemeConfig,
  headTags: [
    // Inter + JetBrains Mono. Preconnect first so the font CSS and the font
    // files resolve in parallel rather than in a serial @import waterfall.
    {
      tagName: 'link',
      attributes: { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
    },
    {
      tagName: 'link',
      attributes: { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: 'anonymous' },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'stylesheet',
        href: 'https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400..700&family=JetBrains+Mono:wght@400;500;600&display=swap',
      },
    },
    // The homepage's Cormorant faces. Not in the homepage's <Head>: it re-inserts its tags on hydration, and the faces
    // would vanish for a few frames. The files load only where the family is used.
    {
      tagName: 'link',
      attributes: { rel: 'stylesheet', href: '/fonts/cormorant.css' },
    },
    {
      tagName: 'script',
      attributes: { type: 'application/ld+json' },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'SoftwareApplication',
        'name': 'vLLM Semantic Router',
        'applicationCategory': 'AIInfrastructure',
        'operatingSystem': 'Cross-platform',
        'description': siteDefaultDescription,
        'url': 'https://vllm-sr.ai',
        'publisher': {
          '@type': 'Organization',
          'name': 'vLLM Semantic Router Team',
          'url': 'https://github.com/vllm-project/semantic-router',
        },
      }),
    },
  ],
}

export default config
