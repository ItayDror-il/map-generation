import React, { useState, useEffect } from 'react';

const LOADING_MESSAGES = [
  'Analyzing your financial profile...',
  'Designing your automation rules...',
  'Optimizing money flow patterns...',
  'Building your personalized map...',
  'Almost there...',
];

export const LoadingState: React.FC = () => {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) =>
        prev < LOADING_MESSAGES.length - 1 ? prev + 1 : prev
      );
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="loading-container" role="status" aria-live="polite">
      <div className="loading-orb" aria-hidden="true" />
      <p className="loading-text">{LOADING_MESSAGES[messageIndex]}</p>
      <p className="loading-subtext">
        This usually takes 10-20 seconds
      </p>
    </div>
  );
};
