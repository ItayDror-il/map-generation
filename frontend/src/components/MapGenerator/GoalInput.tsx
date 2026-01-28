import React from 'react';

interface GoalInputProps {
  value: string;
  onChange: (value: string) => void;
  maxLength?: number;
  minLength?: number;
}

export const GoalInput: React.FC<GoalInputProps> = ({
  value,
  onChange,
  maxLength = 1000,
  minLength = 20,
}) => {
  const charCount = value.length;
  const isNearLimit = charCount > maxLength * 0.9;
  const isTooShort = charCount > 0 && charCount < minLength;

  return (
    <div className="field-group">
      <label htmlFor="goalDescription" className="field-label">
        Describe your financial goals <span className="field-required">*</span>
      </label>
      <div className="textarea-wrapper">
        <textarea
          id="goalDescription"
          className={`textarea-input ${isTooShort ? 'has-error' : ''}`}
          value={value}
          onChange={(e) => onChange(e.target.value.slice(0, maxLength))}
          placeholder="I want to pay off my credit cards while building an emergency fund. I'd like to save 15% of each paycheck and put extra money toward my highest interest debt first..."
          minLength={minLength}
          maxLength={maxLength}
          required
          aria-describedby="goal-hint goal-counter"
        />
        <span
          id="goal-counter"
          className={`textarea-counter ${isNearLimit ? 'near-limit' : ''}`}
          aria-live="polite"
        >
          {charCount}/{maxLength}
        </span>
      </div>
      {isTooShort && (
        <p className="field-error" role="alert">
          <ErrorIcon />
          Please add at least {minLength - charCount} more characters
        </p>
      )}
      <p id="goal-hint" className="text-caption" style={{ marginTop: 'var(--space-2)' }}>
        Be specific about amounts, accounts, and priorities for the best results.
      </p>
    </div>
  );
};

const ErrorIcon: React.FC = () => (
  <svg
    width="14"
    height="14"
    viewBox="0 0 14 14"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.5" />
    <path
      d="M7 4V7.5M7 9.5V10"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);
