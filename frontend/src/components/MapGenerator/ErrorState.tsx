import React from 'react';

interface ErrorStateProps {
  message: string;
  onRetry: () => void;
  onReset: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  message,
  onRetry,
  onReset,
}) => {
  return (
    <div className="error-container" role="alert">
      <div className="error-icon" aria-hidden="true">
        <span role="img" aria-label="error">😕</span>
      </div>
      <h2 className="error-title">Something went wrong</h2>
      <p className="error-message">{message}</p>

      <div className="results-actions">
        <button
          type="button"
          className="btn btn-primary"
          onClick={onRetry}
          style={{ flex: 1 }}
        >
          <RetryIcon />
          <span>Try Again</span>
        </button>

        <button
          type="button"
          className="btn btn-secondary"
          onClick={onReset}
          style={{ flex: 1 }}
        >
          <span>Start Over</span>
        </button>
      </div>
    </div>
  );
};

const RetryIcon: React.FC = () => (
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
