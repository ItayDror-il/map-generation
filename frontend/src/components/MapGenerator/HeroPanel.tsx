import React from 'react';

export const HeroPanel: React.FC = () => {
  return (
    <aside className="hero-panel" aria-hidden="true">
      {/* Animated connection lines */}
      <svg className="hero-connections" viewBox="0 0 400 400">
        <path
          className="hero-connection"
          d="M 80 120 Q 200 80 240 160"
        />
        <path
          className="hero-connection"
          d="M 240 160 Q 280 220 200 280"
        />
        <path
          className="hero-connection"
          d="M 200 280 Q 120 320 80 260"
        />
        <path
          className="hero-connection"
          d="M 80 260 Q 40 180 80 120"
        />
        <path
          className="hero-connection"
          d="M 200 200 L 80 120"
        />
        <path
          className="hero-connection"
          d="M 200 200 L 240 160"
        />
        <path
          className="hero-connection"
          d="M 200 200 L 200 280"
        />
        <path
          className="hero-connection"
          d="M 200 200 L 80 260"
        />
      </svg>

      {/* Floating node decorations */}
      <div className="hero-nodes">
        <div className="hero-node" aria-hidden="true">
          <span role="img" aria-label="income">📥</span>
        </div>
        <div className="hero-node" aria-hidden="true">
          <span role="img" aria-label="savings">🏦</span>
        </div>
        <div className="hero-node" aria-hidden="true">
          <span role="img" aria-label="investment">📈</span>
        </div>
        <div className="hero-node" aria-hidden="true">
          <span role="img" aria-label="card">💳</span>
        </div>
        <div className="hero-node" aria-hidden="true">
          <span role="img" aria-label="money">💰</span>
        </div>
      </div>

      {/* Hero text */}
      <div className="hero-content">
        <h2 className="hero-title">
          Automate Your<br />Money Flow
        </h2>
        <p className="hero-subtitle">
          Set rules once. Let your money move itself to the right places,
          every single time.
        </p>
      </div>
    </aside>
  );
};
