import type { ReactNode } from 'react'
import Link from '@docusaurus/Link'
import Layout from '@theme/Layout'
import React from 'react'
import { committerActivityWindow } from '@site/src/data/committerActivity.generated'
import styles from './governance.module.css'

const roles = [
  {
    index: '01',
    title: 'Contributor',
    summary: 'Participates through code, docs, research, issues, and community work.',
    authority: 'No standing repository authority.',
  },
  {
    index: '02',
    title: 'Committer',
    summary: 'A trusted, active contributor with sustained ownership and review duties.',
    authority: 'Scoped review, triage, and merge responsibility.',
  },
  {
    index: '03',
    title: 'Maintainer',
    summary: 'A project-wide technical and operational steward.',
    authority: 'Release, access, architecture, and final technical responsibility.',
  },
]

export default function Governance(): ReactNode {
  return (
    <Layout
      title="Community Roles & Governance"
      description="Roles, promotion, duties, and emeritus policy for vLLM Semantic Router"
    >
      <div className={styles.container}>
        <header className={styles.hero}>
          <span className={styles.eyebrow}>Community / Governance</span>
          <div className={styles.heroGrid}>
            <h1>Roles & Governance</h1>
            <div>
              <p>
                A clear path from contribution to project stewardship, with
                active responsibility separated from long-term recognition.
              </p>
              <Link to="/community/team">
                View the current team
                <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </header>

        <main className={styles.main}>
          <GovernanceSection index="01" title="Operating principles">
            <div className={styles.principles}>
              <Principle title="Earned authority">
                Permissions follow demonstrated responsibility, judgement, and
                sustained participation—not job title or employer.
              </Principle>
              <Principle title="Public by default">
                Promotions, policy changes, and project-wide decisions are
                recorded in issues or pull requests whenever possible.
              </Principle>
              <Principle title="Active stewardship">
                Elevated access belongs to people familiar with the current
                project; emeritus status preserves credit without implying
                active responsibility.
              </Principle>
            </div>
          </GovernanceSection>

          <GovernanceSection index="02" title="Technical role ladder">
            <p className={styles.sectionIntro}>
              Committer and Maintainer form the repository responsibility
              ladder. Steering Committee membership is separate and does not
              automatically grant code authority.
            </p>
            <div className={styles.roleGrid}>
              {roles.map(role => (
                <article className={styles.roleCard} key={role.title}>
                  <span>{role.index}</span>
                  <h3>{role.title}</h3>
                  <p>{role.summary}</p>
                  <strong>{role.authority}</strong>
                </article>
              ))}
            </div>

            <div className={styles.tableWrap}>
              <table className={styles.roleTable}>
                <thead>
                  <tr>
                    <th>Role</th>
                    <th>Core duties</th>
                    <th>Expected response</th>
                    <th>Standing authority</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th>Committer</th>
                    <td>Review correctness, triage issues, own accepted changes, mentor contributors.</td>
                    <td>Respond to assigned reviews and issues; communicate periods of absence.</td>
                    <td>Review and merge within trusted areas when required checks and approvals pass.</td>
                  </tr>
                  <tr>
                    <th>Maintainer</th>
                    <td>Set engineering direction, approve releases, steward access, resolve escalations.</td>
                    <td>Maintain project-wide review coverage and incident/release continuity.</td>
                    <td>Final technical approval, repository administration, release and security coordination.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </GovernanceSection>

          <GovernanceSection index="03" title="Promotion">
            <div className={styles.promotionGrid}>
              <PromotionCard
                from="Contributor"
                to="Committer"
                requirements={[
                  'Sustained participation for at least three months.',
                  'At least five substantial merged PRs, or equivalent research, documentation, or community work.',
                  'At least five meaningful reviews or issue-triage actions that demonstrate sound judgement.',
                  'A Maintainer sponsor and a public nomination PR with a seven-day review window.',
                ]}
                decision="Approved by a majority of active Maintainers with no unresolved technical or conduct objection."
              />
              <PromotionCard
                from="Committer"
                to="Maintainer"
                requirements={[
                  'Active Committer for at least three months.',
                  'Demonstrated ownership of a subsystem, release function, or cross-project responsibility.',
                  'Primary reviewer for at least ten substantial PRs and reviewer or author of at least twenty substantial changes.',
                  'A record of mentoring, reliable judgement, and handling difficult project-wide tradeoffs.',
                ]}
                decision="Nominated by a Maintainer and approved by at least two-thirds of active Maintainers, with the decision recorded publicly."
              />
            </div>
            <p className={styles.policyNote}>
              Numeric thresholds are minimum evidence, not an automatic grant.
              Quality, breadth, collaboration, and conduct remain decisive.
            </p>
          </GovernanceSection>

          <GovernanceSection index="04" title="Steering Committee">
            <div className={styles.committeeGrid}>
              <div>
                <h3>Purpose</h3>
                <p>
                  The Steering Committee owns long-range project direction,
                  governance policy, cross-track alignment, and final
                  non-technical escalation. It delegates day-to-day engineering
                  to Maintainers.
                </p>
              </div>
              <div>
                <h3>Composition</h3>
                <p>
                  One committee is organized into Industry and Academic tracks.
                  Track membership expresses the perspective represented; it is
                  not a separate hierarchy.
                </p>
              </div>
              <div>
                <h3>Selection</h3>
                <p>
                  Candidates must demonstrate sustained leadership and
                  cross-track collaboration. A current member nominates the
                  candidate, conflicts are disclosed, and a majority of the
                  committee approves the appointment in a recorded decision.
                </p>
              </div>
              <div>
                <h3>Duties</h3>
                <p>
                  Maintain project mission and values, evaluate research and
                  industry alignment, review governance health, and resolve
                  questions that cannot be delegated cleanly.
                </p>
              </div>
            </div>
            <Link className={styles.inlineLink} to="/community/steering-committee">
              Meet the Steering Committee
              <span aria-hidden="true">→</span>
            </Link>
          </GovernanceSection>

          <GovernanceSection index="05" title="Emeritus policy">
            <div className={styles.emeritusLead}>
              <div>
                <span>Rolling window</span>
                <strong>
                  {committerActivityWindow.months}
                  {' '}
                  months
                </strong>
              </div>
              <p>
                An active Committer with no qualifying activity across a rolling
                three-calendar-month window is automatically listed as an
                Emeritus Committer on the next roster refresh.
              </p>
            </div>

            <div className={styles.emeritusGrid}>
              <PolicyStep index="01" title="Measure">
                Count authored PRs, submitted GitHub reviews, and issues either
                authored or commented on in the semantic-router repository.
              </PolicyStep>
              <PolicyStep index="02" title="Reclassify">
                When all three counts are zero, the generated roster moves the
                person to Emeritus while preserving their profile and credit.
              </PolicyStep>
              <PolicyStep index="03" title="Confirm access">
                Maintainers review non-code exceptions and separately reconcile
                repository permissions; the website does not mutate GitHub access.
              </PolicyStep>
              <PolicyStep index="04" title="Return">
                An Emeritus Committer may return after renewed qualifying work
                and Maintainer confirmation that they are current with the project.
              </PolicyStep>
            </div>
            <p className={styles.auditWindow}>
              Current audit window:
              {' '}
              <time dateTime={committerActivityWindow.cutoffDate}>
                {committerActivityWindow.cutoffDate}
              </time>
              {' — '}
              <time dateTime={committerActivityWindow.generatedAt}>
                {committerActivityWindow.generatedAt}
              </time>
              . Non-code contributions may be documented for a Maintainer-reviewed exception.
            </p>
          </GovernanceSection>

          <GovernanceSection index="06" title="Design references">
            <p className={styles.sectionIntro}>
              This model adapts Kubernetes community practices to a smaller,
              single-project community and intentionally uses the shorter
              three-month inactivity window selected for vLLM Semantic Router.
            </p>
            <div className={styles.references}>
              <a
                href="https://github.com/kubernetes/community/blob/main/community-membership.md"
                target="_blank"
                rel="noreferrer"
              >
                <strong>Kubernetes community membership</strong>
                <span>Role progression, review duties, and inactive membership ↗</span>
              </a>
              <a
                href="https://github.com/kubernetes/community/blob/main/contributors/guide/owners.md"
                target="_blank"
                rel="noreferrer"
              >
                <strong>Kubernetes OWNERS guide</strong>
                <span>Scoped review and approval responsibility ↗</span>
              </a>
              <a
                href="https://github.com/kubernetes/steering/blob/main/charter.md"
                target="_blank"
                rel="noreferrer"
              >
                <strong>Kubernetes Steering Committee charter</strong>
                <span>Delegation, governance, voting, and escalation ↗</span>
              </a>
            </div>
          </GovernanceSection>
        </main>
      </div>
    </Layout>
  )
}

function GovernanceSection({
  index,
  title,
  children,
}: {
  index: string
  title: string
  children: ReactNode
}): ReactNode {
  return (
    <section className={styles.section} aria-labelledby={`governance-${index}`}>
      <header className={styles.sectionHeader}>
        <span>{index}</span>
        <h2 id={`governance-${index}`}>{title}</h2>
      </header>
      <div className={styles.sectionBody}>{children}</div>
    </section>
  )
}

function Principle({
  title,
  children,
}: {
  title: string
  children: ReactNode
}): ReactNode {
  return (
    <article>
      <h3>{title}</h3>
      <p>{children}</p>
    </article>
  )
}

function PromotionCard({
  from,
  to,
  requirements,
  decision,
}: {
  from: string
  to: string
  requirements: string[]
  decision: string
}): ReactNode {
  return (
    <article className={styles.promotionCard}>
      <header>
        <span>{from}</span>
        <span aria-hidden="true">→</span>
        <strong>{to}</strong>
      </header>
      <ol>
        {requirements.map(requirement => (
          <li key={requirement}>{requirement}</li>
        ))}
      </ol>
      <p>
        <strong>Decision:</strong>
        {' '}
        {decision}
      </p>
    </article>
  )
}

function PolicyStep({
  index,
  title,
  children,
}: {
  index: string
  title: string
  children: ReactNode
}): ReactNode {
  return (
    <article>
      <span>{index}</span>
      <h3>{title}</h3>
      <p>{children}</p>
    </article>
  )
}
