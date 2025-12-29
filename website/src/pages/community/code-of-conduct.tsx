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
                We as members, contributors, and leaders pledge to make participation in our
                community a harassment-free experience for everyone, regardless of age, body
                size, visible or invisible disability, ethnicity, sex characteristics, gender
                identity and expression, level of experience, education, socio-economic status,
                nationality, personal appearance, race, religion, or sexual identity
                and orientation.
              </p>
              <p>
                We pledge to act and interact in ways that contribute to an open, welcoming,
                diverse, inclusive, and healthy community.
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
                <h3>Examples of behavior that contributes to a positive environment:</h3>
                <ul>
                  <li>Demonstrating empathy and kindness toward other people</li>
                  <li>Being respectful of differing opinions, viewpoints, and experiences</li>
                  <li>Giving and gracefully accepting constructive feedback</li>
                  <li>Accepting responsibility and apologizing to those affected by our mistakes</li>
                  <li>Focusing on what is best not just for us as individuals, but for the overall community</li>
                </ul>
              </div>

              <div className={styles.card}>
                <h3>Examples of unacceptable behavior:</h3>
                <ul>
                  <li>The use of sexualized language or imagery, and sexual attention or advances of any kind</li>
                  <li>Trolling, insulting or derogatory comments, and personal or political attacks</li>
                  <li>Public or private harassment</li>
                  <li>Publishing others' private information without their explicit permission</li>
                  <li>Other conduct which could reasonably be considered inappropriate in a professional setting</li>
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
                Community leaders are responsible for clarifying and enforcing our standards of
                acceptable behavior and will take appropriate and fair corrective action in
                response to any behavior that they deem inappropriate, threatening, offensive,
                or harmful.
              </p>
              <p>
                Community leaders have the right and responsibility to remove, edit, or reject
                comments, commits, code, wiki edits, issues, and other contributions that are
                not aligned to this Code of Conduct, and will communicate reasons for moderation
                decisions when appropriate.
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
                This Code of Conduct applies within all community spaces, and also applies when
                an individual is officially representing the community in public spaces.
                Examples of representing our community include using an official e-mail address,
                posting via an official social media account, or acting as an appointed
                representative at an online or offline event.
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
                Instances of abusive, harassing, or otherwise unacceptable behavior may be
                reported to the community leaders responsible for enforcement through:
              </p>
              <ul>
                <li>
                  <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer">GitHub Issues</a>
                  {' '}
                  (for public issues)
                </li>
                <li>Direct contact with project maintainers</li>
                <li>Email to the project team</li>
              </ul>
              <p>
                All complaints will be reviewed and investigated promptly and fairly.
                All community leaders are obligated to respect the privacy and security of the
                reporter of any incident.
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
                Community leaders will follow these Community Impact Guidelines in determining
                the consequences for any action they deem in violation of this Code of Conduct:
              </p>

              <div className={styles.enforcementGrid}>
                <div className={styles.enforcementItem}>
                  <h4>1. Correction</h4>
                  <p>
                    <strong>Community Impact:</strong>
                    {' '}
                    Use of inappropriate language or other behavior deemed unprofessional or unwelcome in the community.
                  </p>
                  <p>
                    <strong>Consequence:</strong>
                    {' '}
                    A private, written warning from community leaders, providing clarity around the nature of the violation and an explanation of why the behavior was inappropriate.
                  </p>
                </div>

                <div className={styles.enforcementItem}>
                  <h4>2. Warning</h4>
                  <p>
                    <strong>Community Impact:</strong>
                    {' '}
                    A violation through a single incident or series of actions.
                  </p>
                  <p>
                    <strong>Consequence:</strong>
                    {' '}
                    A warning with consequences for continued behavior. No interaction with the people involved for a specified period of time.
                  </p>
                </div>

                <div className={styles.enforcementItem}>
                  <h4>3. Temporary Ban</h4>
                  <p>
                    <strong>Community Impact:</strong>
                    {' '}
                    A serious violation of community standards, including sustained inappropriate behavior.
                  </p>
                  <p>
                    <strong>Consequence:</strong>
                    {' '}
                    A temporary ban from any sort of interaction or public communication with the community for a specified period of time.
                  </p>
                </div>

                <div className={styles.enforcementItem}>
                  <h4>4. Permanent Ban</h4>
                  <p>
                    <strong>Community Impact:</strong>
                    {' '}
                    Demonstrating a pattern of violation of community standards, including sustained inappropriate behavior, harassment of an individual, or aggression toward or disparagement of classes of individuals.
                  </p>
                  <p>
                    <strong>Consequence:</strong>
                    {' '}
                    A permanent ban from any sort of public interaction within the community.
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
                This Code of Conduct is adapted from the
                {' '}
                <a href="https://www.contributor-covenant.org/" target="_blank" rel="noopener noreferrer">Contributor Covenant</a>
                ,
                version 2.0, available at
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
