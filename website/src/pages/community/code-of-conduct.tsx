import React from 'react'
import Layout from '@theme/Layout'
import Translate from '@docusaurus/Translate'
import styles from './community-page.module.css'

const CodeOfConduct: React.FC = () => {
  return (
    <Layout
      title="Code of Conduct"
      description="vLLM Semantic Router Community Code of Conduct"
    >
      <div className={styles.container}>
        <header className={styles.header}>
          <h1><Translate id="codeOfConduct.title">Code of Conduct 📜</Translate></h1>
          <p className={styles.subtitle}>
            <Translate id="codeOfConduct.subtitle">Our commitment to fostering an open, welcoming, and inclusive community.</Translate>
          </p>
        </header>

        <main className={styles.main}>
          <section className={styles.section}>
            <h2>
              🤝
              <Translate id="codeOfConduct.pledge.title">Our Pledge</Translate>
            </h2>
            <div className={styles.card}>
              <p>
                <Translate id="codeOfConduct.pledge.desc1">
                  We as members, contributors, and leaders pledge to make participation in our
                  community a harassment-free experience for everyone, regardless of age, body
                  size, visible or invisible disability, ethnicity, sex characteristics, gender
                  identity and expression, level of experience, education, socio-economic status,
                  nationality, personal appearance, race, religion, or sexual identity
                  and orientation.
                </Translate>
              </p>
              <p>
                <Translate id="codeOfConduct.pledge.desc2">
                  We pledge to act and interact in ways that contribute to an open, welcoming,
                  diverse, inclusive, and healthy community.
                </Translate>
              </p>
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              ✅
              <Translate id="codeOfConduct.standards.title">Our Standards</Translate>
            </h2>
            <div className={styles.grid}>
              <div className={styles.card}>
                <h3><Translate id="codeOfConduct.standards.positive.title">Examples of behavior that contributes to a positive environment:</Translate></h3>
                <ul>
                  <li><Translate id="codeOfConduct.standards.positive.list.1">Demonstrating empathy and kindness toward other people</Translate></li>
                  <li><Translate id="codeOfConduct.standards.positive.list.2">Being respectful of differing opinions, viewpoints, and experiences</Translate></li>
                  <li><Translate id="codeOfConduct.standards.positive.list.3">Giving and gracefully accepting constructive feedback</Translate></li>
                  <li><Translate id="codeOfConduct.standards.positive.list.4">Accepting responsibility and apologizing to those affected by our mistakes</Translate></li>
                  <li><Translate id="codeOfConduct.standards.positive.list.5">Focusing on what is best not just for us as individuals, but for the overall community</Translate></li>
                </ul>
              </div>

              <div className={styles.card}>
                <h3><Translate id="codeOfConduct.standards.unacceptable.title">Examples of unacceptable behavior:</Translate></h3>
                <ul>
                  <li><Translate id="codeOfConduct.standards.unacceptable.list.1">The use of sexualized language or imagery, and sexual attention or advances of any kind</Translate></li>
                  <li><Translate id="codeOfConduct.standards.unacceptable.list.2">Trolling, insulting or derogatory comments, and personal or political attacks</Translate></li>
                  <li><Translate id="codeOfConduct.standards.unacceptable.list.3">Public or private harassment</Translate></li>
                  <li><Translate id="codeOfConduct.standards.unacceptable.list.4">Publishing others' private information without their explicit permission</Translate></li>
                  <li><Translate id="codeOfConduct.standards.unacceptable.list.5">Other conduct which could reasonably be considered inappropriate in a professional setting</Translate></li>
                </ul>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              🛡️
              <Translate id="codeOfConduct.enforcement.title">Enforcement Responsibilities</Translate>
            </h2>
            <div className={styles.card}>
              <p>
                <Translate id="codeOfConduct.enforcement.desc1">
                  Community leaders are responsible for clarifying and enforcing our standards of
                  acceptable behavior and will take appropriate and fair corrective action in
                  response to any behavior that they deem inappropriate, threatening, offensive,
                  or harmful.
                </Translate>
              </p>
              <p>
                <Translate id="codeOfConduct.enforcement.desc2">
                  Community leaders have the right and responsibility to remove, edit, or reject
                  comments, commits, code, wiki edits, issues, and other contributions that are
                  not aligned to this Code of Conduct, and will communicate reasons for moderation
                  decisions when appropriate.
                </Translate>
              </p>
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              🌍
              <Translate id="codeOfConduct.scope.title">Scope</Translate>
            </h2>
            <div className={styles.card}>
              <p>
                <Translate id="codeOfConduct.scope.desc">
                  This Code of Conduct applies within all community spaces, and also applies when
                  an individual is officially representing the community in public spaces.
                  Examples of representing our community include using an official e-mail address,
                  posting via an official social media account, or acting as an appointed
                  representative at an online or offline event.
                </Translate>
              </p>
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              📢
              <Translate id="codeOfConduct.reporting.title">Reporting</Translate>
            </h2>
            <div className={styles.card}>
              <p>
                <Translate id="codeOfConduct.reporting.desc1">
                  Instances of abusive, harassing, or otherwise unacceptable behavior may be
                  reported to the community leaders responsible for enforcement through:
                </Translate>
              </p>
              <ul>
                <li>
                  <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer">GitHub Issues</a>
                  {' '}
                  <Translate id="codeOfConduct.reporting.list.1">(for public issues)</Translate>
                </li>
                <li><Translate id="codeOfConduct.reporting.list.2">Direct contact with project maintainers</Translate></li>
                <li><Translate id="codeOfConduct.reporting.list.3">Email to the project team</Translate></li>
              </ul>
              <p>
                <Translate id="codeOfConduct.reporting.desc2">
                  All complaints will be reviewed and investigated promptly and fairly.
                  All community leaders are obligated to respect the privacy and security of the
                  reporter of any incident.
                </Translate>
              </p>
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              ⚖️
              <Translate id="codeOfConduct.guidelines.title">Enforcement Guidelines</Translate>
            </h2>
            <div className={styles.card}>
              <p>
                <Translate id="codeOfConduct.guidelines.desc">
                  Community leaders will follow these Community Impact Guidelines in determining
                  the consequences for any action they deem in violation of this Code of Conduct:
                </Translate>
              </p>

              <div className={styles.enforcementGrid}>
                <div className={styles.enforcementItem}>
                  <h4><Translate id="codeOfConduct.guidelines.1.title">1. Correction</Translate></h4>
                  <p>
                    <strong><Translate id="codeOfConduct.guidelines.impact">Community Impact:</Translate></strong>
                    {' '}
                    <Translate id="codeOfConduct.guidelines.1.impact">Use of inappropriate language or other behavior deemed unprofessional or unwelcome in the community.</Translate>
                  </p>
                  <p>
                    <strong><Translate id="codeOfConduct.guidelines.consequence">Consequence:</Translate></strong>
                    {' '}
                    <Translate id="codeOfConduct.guidelines.1.consequence">A private, written warning from community leaders, providing clarity around the nature of the violation and an explanation of why the behavior was inappropriate.</Translate>
                  </p>
                </div>

                <div className={styles.enforcementItem}>
                  <h4><Translate id="codeOfConduct.guidelines.2.title">2. Warning</Translate></h4>
                  <p>
                    <strong><Translate id="codeOfConduct.guidelines.impact">Community Impact:</Translate></strong>
                    {' '}
                    <Translate id="codeOfConduct.guidelines.2.impact">A violation through a single incident or series of actions.</Translate>
                  </p>
                  <p>
                    <strong><Translate id="codeOfConduct.guidelines.consequence">Consequence:</Translate></strong>
                    {' '}
                    <Translate id="codeOfConduct.guidelines.2.consequence">A warning with consequences for continued behavior. No interaction with the people involved for a specified period of time.</Translate>
                  </p>
                </div>

                <div className={styles.enforcementItem}>
                  <h4><Translate id="codeOfConduct.guidelines.3.title">3. Temporary Ban</Translate></h4>
                  <p>
                    <strong><Translate id="codeOfConduct.guidelines.impact">Community Impact:</Translate></strong>
                    {' '}
                    <Translate id="codeOfConduct.guidelines.3.impact">A serious violation of community standards, including sustained inappropriate behavior.</Translate>
                  </p>
                  <p>
                    <strong><Translate id="codeOfConduct.guidelines.consequence">Consequence:</Translate></strong>
                    {' '}
                    <Translate id="codeOfConduct.guidelines.3.consequence">A temporary ban from any sort of interaction or public communication with the community for a specified period of time.</Translate>
                  </p>
                </div>

                <div className={styles.enforcementItem}>
                  <h4><Translate id="codeOfConduct.guidelines.4.title">4. Permanent Ban</Translate></h4>
                  <p>
                    <strong><Translate id="codeOfConduct.guidelines.impact">Community Impact:</Translate></strong>
                    {' '}
                    <Translate id="codeOfConduct.guidelines.4.impact">Demonstrating a pattern of violation of community standards, including sustained inappropriate behavior, harassment of an individual, or aggression toward or disparagement of classes of individuals.</Translate>
                  </p>
                  <p>
                    <strong><Translate id="codeOfConduct.guidelines.consequence">Consequence:</Translate></strong>
                    {' '}
                    <Translate id="codeOfConduct.guidelines.4.consequence">A permanent ban from any sort of public interaction within the community.</Translate>
                  </p>
                </div>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              📚
              <Translate id="codeOfConduct.attribution.title">Attribution</Translate>
            </h2>
            <div className={styles.card}>
              <p>
                <Translate id="codeOfConduct.attribution.desc.prefix">This Code of Conduct is adapted from the</Translate>
                {' '}
                <a href="https://www.contributor-covenant.org/" target="_blank" rel="noopener noreferrer">Contributor Covenant</a>
                ,
                <Translate id="codeOfConduct.attribution.desc.middle">version 2.0, available at</Translate>
                {' '}
                <a href="https://www.contributor-covenant.org/version/2/0/code_of_conduct.html" target="_blank" rel="noopener noreferrer">https://www.contributor-covenant.org/version/2/0/code_of_conduct.html</a>
                .
              </p>
            </div>
          </section>
        </main>
      </div>
    </Layout>
  )
}

export default CodeOfConduct
