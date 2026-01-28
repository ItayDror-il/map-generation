import React, { useState } from 'react';
import { ProfileForm } from './ProfileForm';
import { GoalInput } from './GoalInput';
import { GenerateButton } from './GenerateButton';
import { ResultsDisplay } from './ResultsDisplay';
import { LoadingState } from './LoadingState';
import { ErrorState } from './ErrorState';
import { HeroPanel } from './HeroPanel';
import { useMapGenerator } from '../../hooks/useMapGenerator';
import { ProfileFormData } from '../../types';
import '../../styles/design-system.css';
import '../../styles/components.css';

type ViewState = 'form' | 'loading' | 'results' | 'error';

const INITIAL_FORM_DATA: ProfileFormData = {
  userType: '' as 'INDIVIDUAL' | 'BUSINESS',
  annualIncome: '',
  ageGroup: '',
  occupation: '',
  productGoals: [],
  goalDescription: '',
};

export const MapGenerator: React.FC = () => {
  const [viewState, setViewState] = useState<ViewState>('form');
  const [formData, setFormData] = useState<ProfileFormData>(INITIAL_FORM_DATA);

  const {
    generateMap,
    result,
    error,
    isRateLimited,
    rateLimitCountdown,
    reset: resetHook,
  } = useMapGenerator();

  const handleSubmit = async () => {
    setViewState('loading');
    try {
      await generateMap(formData);
      setViewState('results');
    } catch {
      setViewState('error');
    }
  };

  const handleReset = () => {
    setFormData(INITIAL_FORM_DATA);
    resetHook();
    setViewState('form');
  };

  const handleRetry = () => {
    handleSubmit();
  };

  const isFormValid =
    formData.userType &&
    formData.annualIncome &&
    formData.ageGroup &&
    formData.occupation &&
    formData.goalDescription.length >= 20;

  return (
    <div className="page-container">
      <HeroPanel />

      <div className="form-panel">
        <div className="form-wrapper">
          {viewState === 'form' && (
            <>
              <header className="form-header">
                <div className="form-badge">
                  <span className="form-badge-dot" />
                  <span>AI-Powered</span>
                </div>
                <h1 className="form-title">
                  Design Your Financial Automation
                </h1>
                <p className="form-description">
                  Tell us about yourself and your goals. We'll create a
                  personalized money flow map in seconds.
                </p>
              </header>

              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  if (isFormValid && !isRateLimited) handleSubmit();
                }}
              >
                <div className="form-fields">
                  <ProfileForm
                    formData={formData}
                    onChange={setFormData}
                  />

                  <GoalInput
                    value={formData.goalDescription}
                    onChange={(value) =>
                      setFormData((prev) => ({
                        ...prev,
                        goalDescription: value,
                      }))
                    }
                  />

                  <GenerateButton
                    disabled={!isFormValid}
                    isRateLimited={isRateLimited}
                    rateLimitCountdown={rateLimitCountdown}
                  />
                </div>
              </form>
            </>
          )}

          {viewState === 'loading' && <LoadingState />}

          {viewState === 'results' && result && (
            <ResultsDisplay
              explanation={result.explanation}
              mapId={result.id}
              mapJson={result.map}
              onReset={handleReset}
            />
          )}

          {viewState === 'error' && (
            <ErrorState
              message={error || 'Something went wrong'}
              onRetry={handleRetry}
              onReset={handleReset}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default MapGenerator;
