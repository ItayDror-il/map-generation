import React from 'react';
import { ProfileFormData, ProductGoal } from '../../types';

interface ProfileFormProps {
  formData: ProfileFormData;
  onChange: (data: ProfileFormData) => void;
}

const USER_TYPES = [
  { value: 'INDIVIDUAL', label: 'Individual' },
  { value: 'BUSINESS', label: 'Business' },
];

const INCOME_RANGES = [
  { value: 'UNDER_25K', label: 'Under $25,000' },
  { value: 'BETWEEN_25K_AND_50K', label: '$25,000 - $50,000' },
  { value: 'BETWEEN_50K_AND_100K', label: '$50,000 - $100,000' },
  { value: 'BETWEEN_100K_AND_250K', label: '$100,000 - $250,000' },
  { value: 'OVER_250K', label: 'Over $250,000' },
];

const AGE_GROUPS = [
  { value: '18-24', label: '18-24' },
  { value: '25-34', label: '25-34' },
  { value: '35-44', label: '35-44' },
  { value: '45-54', label: '45-54' },
  { value: '55-64', label: '55-64' },
  { value: '65+', label: '65+' },
];

const OCCUPATIONS = [
  { value: 'EMPLOYED', label: 'Employed' },
  { value: 'SELF_EMPLOYED', label: 'Self-Employed' },
  { value: 'FREELANCER', label: 'Freelancer' },
  { value: 'GIG_WORKER', label: 'Gig Worker' },
  { value: 'BUSINESS_OWNER', label: 'Business Owner' },
  { value: 'RETIRED', label: 'Retired' },
  { value: 'STUDENT', label: 'Student' },
];

const PRODUCT_GOALS: { value: ProductGoal; label: string; icon: string }[] = [
  { value: 'DEBT_PAYOFF', label: 'Debt', icon: '💳' },
  { value: 'SAVINGS', label: 'Savings', icon: '🏦' },
  { value: 'BUDGETING', label: 'Budget', icon: '📊' },
  { value: 'TAX_PLANNING', label: 'Taxes', icon: '📋' },
  { value: 'INVESTMENT', label: 'Invest', icon: '📈' },
  { value: 'EMERGENCY_FUND', label: 'Emergency', icon: '🛡️' },
];

export const ProfileForm: React.FC<ProfileFormProps> = ({
  formData,
  onChange,
}) => {
  const updateField = <K extends keyof ProfileFormData>(
    key: K,
    value: ProfileFormData[K]
  ) => {
    onChange({ ...formData, [key]: value });
  };

  const toggleGoal = (goal: ProductGoal) => {
    const current = formData.productGoals;
    const updated = current.includes(goal)
      ? current.filter((g) => g !== goal)
      : [...current, goal];
    updateField('productGoals', updated);
  };

  return (
    <>
      {/* User Type & Income Row */}
      <div className="field-row">
        <div className="field-group">
          <label htmlFor="userType" className="field-label">
            I am <span className="field-required">*</span>
          </label>
          <div className="select-wrapper">
            <select
              id="userType"
              className="select-input"
              value={formData.userType}
              onChange={(e) =>
                updateField('userType', e.target.value as 'INDIVIDUAL' | 'BUSINESS')
              }
              required
            >
              <option value="" disabled>
                Select type...
              </option>
              {USER_TYPES.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <ChevronIcon />
          </div>
        </div>

        <div className="field-group">
          <label htmlFor="annualIncome" className="field-label">
            Annual Income <span className="field-required">*</span>
          </label>
          <div className="select-wrapper">
            <select
              id="annualIncome"
              className="select-input"
              value={formData.annualIncome}
              onChange={(e) => updateField('annualIncome', e.target.value)}
              required
            >
              <option value="" disabled>
                Select range...
              </option>
              {INCOME_RANGES.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <ChevronIcon />
          </div>
        </div>
      </div>

      {/* Age & Occupation Row */}
      <div className="field-row">
        <div className="field-group">
          <label htmlFor="ageGroup" className="field-label">
            Age Group <span className="field-required">*</span>
          </label>
          <div className="select-wrapper">
            <select
              id="ageGroup"
              className="select-input"
              value={formData.ageGroup}
              onChange={(e) => updateField('ageGroup', e.target.value)}
              required
            >
              <option value="" disabled>
                Select age...
              </option>
              {AGE_GROUPS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <ChevronIcon />
          </div>
        </div>

        <div className="field-group">
          <label htmlFor="occupation" className="field-label">
            Occupation <span className="field-required">*</span>
          </label>
          <div className="select-wrapper">
            <select
              id="occupation"
              className="select-input"
              value={formData.occupation}
              onChange={(e) => updateField('occupation', e.target.value)}
              required
            >
              <option value="" disabled>
                Select occupation...
              </option>
              {OCCUPATIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <ChevronIcon />
          </div>
        </div>
      </div>

      {/* Product Goals Multi-select */}
      <div className="field-group">
        <label className="field-label">
          What are your goals? <span style={{ fontWeight: 400, color: 'var(--color-charcoal-400)' }}>(optional)</span>
        </label>
        <div className="multiselect-grid" role="group" aria-label="Select your financial goals">
          {PRODUCT_GOALS.map((goal) => (
            <button
              key={goal.value}
              type="button"
              className={`multiselect-option ${
                formData.productGoals.includes(goal.value) ? 'selected' : ''
              }`}
              onClick={() => toggleGoal(goal.value)}
              aria-pressed={formData.productGoals.includes(goal.value)}
            >
              <span className="multiselect-option-icon" aria-hidden="true">
                {goal.icon}
              </span>
              <span>{goal.label}</span>
            </button>
          ))}
        </div>
      </div>
    </>
  );
};

const ChevronIcon: React.FC = () => (
  <svg
    className="select-icon"
    width="16"
    height="16"
    viewBox="0 0 16 16"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      d="M4 6L8 10L12 6"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);
