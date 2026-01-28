import React from 'react';

interface GenerateButtonProps {
  disabled: boolean;
  isRateLimited: boolean;
  rateLimitCountdown: number;
}

export const GenerateButton: React.FC<GenerateButtonProps> = ({
  disabled,
  isRateLimited,
  rateLimitCountdown,
}) => {
  return (
    <div style={{ marginTop: 'var(--space-4)' }}>
      {isRateLimited && (
        <div className="rate-limit-banner" role="alert">
          <span className="rate-limit-icon" aria-hidden="true">
            ⏳
          </span>
          <span className="rate-limit-text">
            Please wait before generating another map
          </span>
          <span className="rate-limit-timer" aria-live="polite">
            {rateLimitCountdown}s
          </span>
        </div>
      )}

      <button
        type="submit"
        className="btn btn-primary"
        disabled={disabled || isRateLimited}
        aria-describedby={isRateLimited ? 'rate-limit-message' : undefined}
      >
        <SparkleIcon />
        <span>Generate My Map</span>
      </button>
    </div>
  );
};

const SparkleIcon: React.FC = () => (
  <svg
    width="20"
    height="20"
    viewBox="0 0 20 20"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M10 2L11.5 7.5L17 9L11.5 10.5L10 16L8.5 10.5L3 9L8.5 7.5L10 2Z"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M15 14L15.75 16L17.5 16.5L15.75 17L15 19L14.25 17L12.5 16.5L14.25 16L15 14Z"
      stroke="currentColor"
      strokeWidth="1.25"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);
