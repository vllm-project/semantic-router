import type { ReactNode } from 'react'
import Translate from '@docusaurus/Translate'
import Layout from '@theme/Layout'
import React from 'react'
import CommunityLayout from '@site/src/components/community/CommunityLayout'
import {
  type WorkGroup,
  type WorkGroupPerson,
  workGroups,
} from '@site/src/data/workGroups'
import styles from './work-groups.module.css'

const GITHUB_BASE = 'https://github.com/vllm-project/semantic-router'
const VLLM_LOGO = '/img/acknowledgements/vllm-logo.png'

export default function WorkGroups(): ReactNode {
  return (
    <Layout
      title="Working Groups"
      description="Direction-based vLLM Semantic Router Working Groups"
    >
      <CommunityLayout
        activeKey="work-groups"
        title={<Translate id="workGroups.page.title">Working Groups</Translate>}
        description={(
          <Translate id="workGroups.page.description">
            Find a technical focus, meet collaborators, and contribute to accepted work.
          </Translate>
        )}
      >
        <div className={styles.main}>
          <section className={styles.groupsSection}>
            <header className={styles.groupsHeader}>
              <h2>
                <Translate id="workGroups.page.focusTitle">Find your focus and passion</Translate>
              </h2>
              <p>
                <Translate id="workGroups.page.focusDescription">
                  Build and grow with the community.
                </Translate>
              </p>
            </header>

            <div className={styles.groupList}>
              {workGroups.map((group, index) => (
                <WorkGroupRow key={group.id} group={group} index={index + 1} />
              ))}
            </div>
          </section>

          <section className={styles.participation}>
            <header className={styles.sectionHeading}>
              <span><Translate id="workGroups.participation.label">Participation</Translate></span>
              <div>
                <h2><Translate id="workGroups.participation.title">Lead and Member roles</Translate></h2>
                <p>
                  <Translate id="workGroups.participation.description">
                    Roles make technical ownership and contribution paths visible.
                  </Translate>
                </p>
              </div>
            </header>
            <dl className={styles.rules}>
              <div>
                <dt><Translate id="workGroups.participation.lead.title">Lead</Translate></dt>
                <dd>
                  <Translate id="workGroups.participation.lead.description">
                    One or more per group; a Committer or a Contributor sponsored by a Maintainer.
                  </Translate>
                </dd>
              </div>
              <div>
                <dt><Translate id="workGroups.participation.member.title">Member</Translate></dt>
                <dd>
                  <Translate id="workGroups.participation.member.description">
                    A Contributor with at least one merged repository commit.
                  </Translate>
                </dd>
              </div>
              <div>
                <dt><Translate id="workGroups.participation.visibility.title">Visibility</Translate></dt>
                <dd>
                  <Translate id="workGroups.participation.visibility.description">
                    Confirmed Leads and Members appear here with their avatar and name.
                  </Translate>
                </dd>
              </div>
              <div>
                <dt><Translate id="workGroups.participation.authority.title">Authority</Translate></dt>
                <dd>
                  <Translate id="workGroups.participation.authority.description">
                    Workgroup roles are separate from Open Source Team repository and release authority.
                  </Translate>
                </dd>
              </div>
            </dl>
          </section>

          <section className={styles.proposal}>
            <p>
              <strong>
                <Translate id="workGroups.proposal.title">Need a new Workgroup?</Translate>
              </strong>
              {' '}
              <Translate id="workGroups.proposal.description">
                Propose one only when a durable problem does not fit an existing charter.
              </Translate>
            </p>
            <a href={`${GITHUB_BASE}/issues/new?template=001_feature_request.yaml`}>
              <Translate id="workGroups.proposal.link">Open a proposal</Translate>
            </a>
          </section>
        </div>
      </CommunityLayout>
    </Layout>
  )
}

function WorkGroupRow({
  group,
  index,
}: {
  group: WorkGroup
  index: number
}): ReactNode {
  const charterUrl = `${GITHUB_BASE}/issues/${group.charterIssue}`

  return (
    <article className={styles.workGroup} id={group.id}>
      <div className={styles.groupLogo}>
        <img src={VLLM_LOGO} alt="" />
      </div>

      <header className={styles.groupIdentity}>
        <span>{String(index).padStart(2, '0')}</span>
        <h3>
          <Translate id={`workGroups.group.${group.id}.name`}>{group.name}</Translate>
        </h3>
        <p>
          <Translate id={`workGroups.group.${group.id}.goal`}>{group.goal}</Translate>
        </p>
        <code>{group.label}</code>
      </header>

      <div className={styles.groupDetails}>
        <div className={styles.focus}>
          <h4><Translate id="workGroups.group.focus">Focus</Translate></h4>
          <ul>
            {group.scope.map((item, scopeIndex) => (
              <li key={item}>
                <Translate id={`workGroups.group.${group.id}.scope.${scopeIndex}`}>{item}</Translate>
              </li>
            ))}
          </ul>
        </div>

        <footer className={styles.groupFooter}>
          <div className={styles.roster}>
            <Roster
              title="Leads"
              titleId="workGroups.roster.leads"
              people={group.leads ?? []}
              empty="Open"
              emptyId="workGroups.roster.open"
            />
            <Roster
              title="Members"
              titleId="workGroups.roster.members"
              people={group.members ?? []}
              empty="Forming"
              emptyId="workGroups.roster.forming"
            />
          </div>
          <a className={styles.charterLink} href={charterUrl}>
            <Translate id="workGroups.group.charterLink">Charter & self-nomination →</Translate>
          </a>
        </footer>
      </div>
    </article>
  )
}

function Roster({
  title,
  titleId,
  people,
  empty,
  emptyId,
}: {
  title: string
  titleId: string
  people: WorkGroupPerson[]
  empty: string
  emptyId: string
}): ReactNode {
  return (
    <div className={styles.rosterGroup}>
      <span><Translate id={titleId}>{title}</Translate></span>
      {people.length === 0
        ? <strong><Translate id={emptyId}>{empty}</Translate></strong>
        : (
            <div className={styles.people}>
              {people.map(person => (
                <a key={person.profile} className={styles.person} href={person.profile}>
                  <img src={person.avatar} alt="" />
                  <strong>{person.name}</strong>
                </a>
              ))}
            </div>
          )}
    </div>
  )
}
