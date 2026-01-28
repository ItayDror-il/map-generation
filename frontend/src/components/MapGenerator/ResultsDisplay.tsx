import React from 'react';

interface ResultsDisplayProps {
  explanation: string;
  mapId: string;
  mapJson?: Record<string, unknown>;
  onReset: () => void;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  explanation,
  mapId,
  mapJson,
  onReset,
}) => {
  const playgroundUrl = `https://playground.getsequence.io/?id=${mapId}`;

  const handleOpenPlayground = () => {
    window.open(playgroundUrl, '_blank', 'noopener,noreferrer');
  };

  return (
    <div className="results-container">
      <div className="results-header">
        <div className="results-icon" aria-hidden="true">
          <span role="img" aria-label="success">✨</span>
        </div>
        <h2 className="results-title">Your Map is Ready</h2>
      </div>

      <div className="results-card">
        <h3
          style={{
            fontFamily: 'var(--font-body)',
            fontWeight: 600,
            fontSize: 'var(--text-base)',
            color: 'var(--color-charcoal-700)',
            marginBottom: 'var(--space-4)',
          }}
        >
          Here's what we built for you:
        </h3>
        <div className="results-explanation" aria-label="Map explanation">
          {explanation}
        </div>
      </div>

      <div className="results-actions">
        <button
          type="button"
          className="btn btn-playground"
          onClick={handleOpenPlayground}
        >
          <ExternalLinkIcon />
          <span>Open in Playground</span>
        </button>

        <button
          type="button"
          className="btn btn-secondary btn-restart"
          onClick={onReset}
        >
          <RefreshIcon />
          <span>Start Over</span>
        </button>
      </div>

      <p
        style={{
          marginTop: 'var(--space-6)',
          fontSize: 'var(--text-sm)',
          color: 'var(--color-charcoal-400)',
          textAlign: 'center',
        }}
      >
        The playground lets you visualize and adjust your map before saving.
      </p>

      {/* Debug: Show raw JSON */}
      {mapJson && (
        <details style={{ marginTop: 'var(--space-6)' }}>
          <summary
            style={{
              cursor: 'pointer',
              fontSize: 'var(--text-sm)',
              color: 'var(--color-charcoal-500)',
            }}
          >
            Debug: View Raw JSON
          </summary>
          <pre
            style={{
              marginTop: 'var(--space-2)',
              padding: 'var(--space-4)',
              background: 'var(--color-charcoal-800)',
              color: 'var(--color-ivory-100)',
              borderRadius: 'var(--radius-md)',
              fontSize: 'var(--text-xs)',
              overflow: 'auto',
              maxHeight: '300px',
            }}
          >
            {JSON.stringify(mapJson, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
};

const ExternalLinkIcon: React.FC = () => (
  <svg
    width="18"
    height="18"
    viewBox="0 0 18 18"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M15 9.75V14.25C15 14.6478 14.842 15.0294 14.5607 15.3107C14.2794 15.592 13.8978 15.75 13.5 15.75H3.75C3.35218 15.75 2.97064 15.592 2.68934 15.3107C2.40804 15.0294 2.25 14.6478 2.25 14.25V4.5C2.25 4.10218 2.40804 3.72064 2.68934 3.43934C2.97064 3.15804 3.35218 3 3.75 3H8.25"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M12 2.25H15.75V6"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M7.5 10.5L15.75 2.25"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const RefreshIcon: React.FC = () => (
  <svg
    width="18"
    height="18"
    viewBox="0 0 18 18"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M1.5 3V7.5H6"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M3.51 11.25C4.01717 12.5765 4.95536 13.6948 6.17316 14.4265C7.39097 15.1582 8.8178 15.4614 10.2265 15.2879C11.6352 15.1145 12.9444 14.4747 13.9485 13.4706C14.9525 12.4665 15.5924 11.1573 15.7658 9.74865C15.9393 8.33995 15.6361 6.91312 14.9044 5.69531C14.1727 4.47751 13.0544 3.53932 11.7279 3.03215C10.4014 2.52498 8.94607 2.43391 7.56407 2.77267C6.18207 3.11143 4.94526 3.8619 4.005 4.92L1.5 7.5"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);
