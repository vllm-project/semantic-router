import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import Translate, { translate } from '@docusaurus/Translate'
import Layout from '@theme/Layout'
import React from 'react'
import CommunityLayout from '@site/src/components/community/CommunityLayout'
import { type PaperChapter, PaperViewer } from '@site/src/components/PaperViewerPage'
import { PillLink } from '@site/src/components/site/Chrome'
import styles from './contributor-guide.module.css'

const PDF_URL = '/vllm-sr-contributor-guide.pdf'
const GITHUB_BASE = 'https://github.com/vllm-project/semantic-router'

// The guide's contents page, with the page on which each chapter opens.
const chapters: PaperChapter[] = [
  {
    page: 3,
    title: translate({ id: 'contributorGuide.chapter.whatItIs', message: 'What vLLM-SR is' }),
    description: translate({ id: 'contributorGuide.chapter.whatItIs.description', message: 'The problem and the goal' }),
  },
  {
    page: 4,
    title: translate({ id: 'contributorGuide.chapter.architecture', message: 'Architecture' }),
    description: translate({ id: 'contributorGuide.chapter.architecture.description', message: 'From one request to the right model' }),
  },
  {
    page: 5,
    title: translate({ id: 'contributorGuide.chapter.moduleDesign', message: 'Module design' }),
    description: translate({ id: 'contributorGuide.chapter.moduleDesign.description', message: 'The routing pipeline and the repository' }),
  },
  {
    page: 6,
    title: translate({ id: 'contributorGuide.chapter.modelFamilies', message: 'Two model families' }),
    description: translate({ id: 'contributorGuide.chapter.modelFamilies.description', message: 'Vela and Decision' }),
  },
  {
    page: 7,
    title: translate({ id: 'contributorGuide.chapter.workGroups', message: 'Work Groups' }),
    description: translate({ id: 'contributorGuide.chapter.workGroups.description', message: 'Seven owners, one system' }),
  },
  {
    page: 10,
    title: translate({ id: 'contributorGuide.chapter.directions', message: 'Directions by priority' }),
    description: translate({ id: 'contributorGuide.chapter.directions.description', message: 'What each Work Group builds next' }),
  },
  {
    page: 11,
    title: translate({ id: 'contributorGuide.chapter.join', message: 'Join a Work Group' }),
    description: translate({ id: 'contributorGuide.chapter.join.description', message: 'Apply, claim, deliver' }),
  },
  {
    page: 12,
    title: translate({ id: 'contributorGuide.chapter.grow', message: 'Grow with the project' }),
    description: translate({ id: 'contributorGuide.chapter.grow.description', message: 'Committer and Maintainer' }),
  },
  {
    page: 13,
    title: translate({ id: 'contributorGuide.chapter.contact', message: 'Get in touch' }),
    description: translate({ id: 'contributorGuide.chapter.contact.description', message: 'Links and community' }),
  },
]

interface GuideLink {
  label: string
  text: string
  href?: string
  to?: string
}

// The guide prints these as plain text; here they are links.
const links: GuideLink[] = [
  {
    label: translate({ id: 'contributorGuide.links.repository', message: 'Repository' }),
    text: 'github.com/vllm-project/semantic-router',
    href: GITHUB_BASE,
  },
  {
    label: translate({ id: 'contributorGuide.links.documentation', message: 'Documentation' }),
    text: 'vllm-sr.ai/docs',
    to: '/docs/intro',
  },
  {
    label: translate({ id: 'contributorGuide.links.workGroups', message: 'Work Groups' }),
    text: 'vllm-sr.ai/community/work-groups',
    to: '/community/work-groups',
  },
  {
    label: translate({ id: 'contributorGuide.links.joinGuide', message: 'Join guide' }),
    text: 'github.com/vllm-project/semantic-router/issues/15',
    href: `${GITHUB_BASE}/issues/15`,
  },
  {
    label: translate({ id: 'contributorGuide.links.weeklyIssues', message: 'Weekly issues' }),
    text: 'community.vllm-sr.ai/weekly-issues',
    href: 'https://community.vllm-sr.ai/weekly-issues',
  },
  {
    label: translate({ id: 'contributorGuide.links.roadmap', message: 'Roadmap survey' }),
    text: 'form.vllm-sr.ai/roadmap',
    href: 'https://form.vllm-sr.ai/roadmap',
  },
  {
    label: translate({ id: 'contributorGuide.links.models', message: 'Models' }),
    text: 'huggingface.co/vllm-sr',
    href: 'https://huggingface.co/vllm-sr',
  },
  {
    label: translate({ id: 'contributorGuide.links.tryIt', message: 'Try it' }),
    text: 'app.vllm-sr.ai',
    href: 'https://app.vllm-sr.ai/',
  },
  {
    label: translate({ id: 'contributorGuide.links.chat', message: 'Chat' }),
    text: translate({ id: 'contributorGuide.links.chat.text', message: '#semantic-router in the vLLM Slack' }),
    href: 'https://vllm-dev.slack.com/archives/C09CTGF8KCN',
  },
  {
    label: translate({ id: 'contributorGuide.links.meetings', message: 'Meetings' }),
    text: translate({
      id: 'contributorGuide.links.meetings.text',
      message: 'Two community meetings a month, for APAC and the Americas',
    }),
    href: `${GITHUB_BASE}#community-meetings`,
  },
]

export default function ContributorGuide(): ReactNode {
  return (
    <Layout
      title={translate({ id: 'contributorGuide.meta.title', message: 'Contributor Guide' })}
      description={translate({
        id: 'contributorGuide.meta.description',
        message: 'How vLLM Semantic Router works, where you can help, and how to grow from your first pull request to maintainer.',
      })}
    >
      <CommunityLayout
        activeKey="contributor-guide"
        title={<Translate id="contributorGuide.page.title">Contributor Guide</Translate>}
        description={(
          <Translate id="contributorGuide.page.description">
            How vLLM Semantic Router works, where you can help, and how to grow from your first pull request to maintainer.
          </Translate>
        )}
        actions={(
          <PillLink href={PDF_URL} target="_blank" rel="noreferrer">
            <Translate id="paperViewer.hero.download">Download PDF</Translate>
          </PillLink>
        )}
      >
        <div className={styles.main}>
          <section className={styles.section} aria-labelledby="contributor-guide-start">
            <header className={styles.sectionHeading}>
              <span><Translate id="contributorGuide.start.label">Start</Translate></span>
              <div>
                <h2 id="contributor-guide-start">
                  <Translate id="contributorGuide.start.title">Three ways to start</Translate>
                </h2>
                <p>
                  <Translate id="contributorGuide.start.description">
                    Try it in your browser, run it on your own machine, or build it with a Work Group.
                  </Translate>
                </p>
              </div>
            </header>

            <ol className={styles.ways}>
              <li>
                <a className={styles.way} href="https://app.vllm-sr.ai/#try" target="_blank" rel="noreferrer">
                  <span className={styles.wayIndex}>01</span>
                  <h3><Translate id="contributorGuide.start.try.title">Try it</Translate></h3>
                  <p>
                    <Translate
                      id="contributorGuide.start.try.description"
                      values={{ site: <code>app.vllm-sr.ai</code> }}
                    >
                      {'Pick an example at {site}. It runs live in your browser, with a free key on first run.'}
                    </Translate>
                  </p>
                </a>
              </li>
              <li>
                <Link className={styles.way} to="/docs/installation">
                  <span className={styles.wayIndex}>02</span>
                  <h3><Translate id="contributorGuide.start.run.title">Run it</Translate></h3>
                  <p>
                    <Translate
                      id="contributorGuide.start.run.description"
                      values={{
                        script: <code>vllm-sr.ai/install.sh</code>,
                        command: <code>vllm-sr serve</code>,
                      }}
                    >
                      {'Install with the script at {script}, then start the local stack with {command}.'}
                    </Translate>
                  </p>
                </Link>
              </li>
              <li>
                <Link className={styles.way} to="/community/work-groups">
                  <span className={styles.wayIndex}>03</span>
                  <h3><Translate id="contributorGuide.start.build.title">Build it</Translate></h3>
                  <p>
                    <Translate id="contributorGuide.start.build.description">
                      Find your Work Group, claim an AVAILABLE issue and open a focused pull request.
                    </Translate>
                  </p>
                </Link>
              </li>
            </ol>
          </section>

          <section className={styles.section} aria-labelledby="contributor-guide-read">
            <header className={styles.sectionHeading}>
              <span><Translate id="contributorGuide.read.label">Guide</Translate></span>
              <div>
                <h2 id="contributor-guide-read">
                  <Translate id="contributorGuide.read.title">Read the guide</Translate>
                </h2>
                <p>
                  <Translate id="contributorGuide.read.description">
                    Edition 1, November 2026. Thirteen pages: jump to a chapter, or turn the pages with the arrow keys.
                  </Translate>
                </p>
              </div>
            </header>

            <PaperViewer pdfUrl={PDF_URL} chapters={chapters} />
          </section>

          <section className={styles.section} aria-labelledby="contributor-guide-contact">
            <header className={styles.sectionHeading}>
              <span><Translate id="contributorGuide.contact.label">Get in touch</Translate></span>
              <div>
                <h2 id="contributor-guide-contact">
                  <Translate id="contributorGuide.contact.title">Let’s build it together</Translate>
                </h2>
                <p>
                  <Translate id="contributorGuide.contact.description">
                    Pick a Work Group, claim a task and say hello. We review every first pull request with care.
                  </Translate>
                </p>
              </div>
            </header>

            <dl className={styles.links}>
              {links.map(link => (
                <div key={link.label}>
                  <dt>{link.label}</dt>
                  <dd>
                    {link.to
                      ? <Link to={link.to}>{link.text}</Link>
                      : <a href={link.href} target="_blank" rel="noreferrer">{link.text}</a>}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        </div>
      </CommunityLayout>
    </Layout>
  )
}
