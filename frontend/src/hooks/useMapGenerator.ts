import { useState, useCallback, useRef, useEffect } from 'react';
import { ProfileFormData } from '../types';

/**
 * API Configuration
 *
 * The API URL is the ONLY thing connecting frontend to backend.
 * Change this to point to any backend implementation.
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Rate limiting configuration
 */
const RATE_LIMIT = {
  maxRequests: 10,
  windowMs: 60 * 1000, // 1 minute
};

/**
 * API Response types - these are the contract with the backend
 */
interface GenerateMapResponse {
  id: string;
  explanation: string;
  map: Record<string, unknown>;
}

interface ApiError {
  detail: string;
}

/**
 * Hook return type
 */
interface UseMapGeneratorReturn {
  generateMap: (formData: ProfileFormData) => Promise<void>;
  result: GenerateMapResponse | null;
  error: string | null;
  isLoading: boolean;
  isRateLimited: boolean;
  rateLimitCountdown: number;
  reset: () => void;
}

/**
 * Map generator hook
 *
 * Handles:
 * - API communication (single endpoint)
 * - Rate limiting (client-side)
 * - Error handling
 * - Loading states
 */
export function useMapGenerator(): UseMapGeneratorReturn {
  const [result, setResult] = useState<GenerateMapResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRateLimited, setIsRateLimited] = useState(false);
  const [rateLimitCountdown, setRateLimitCountdown] = useState(0);

  // Rate limiting state
  const requestTimestamps = useRef<number[]>([]);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
      }
    };
  }, []);

  /**
   * Check if we're rate limited
   */
  const checkRateLimit = useCallback((): boolean => {
    const now = Date.now();
    const windowStart = now - RATE_LIMIT.windowMs;

    // Clean old timestamps
    requestTimestamps.current = requestTimestamps.current.filter(
      (ts) => ts > windowStart
    );

    if (requestTimestamps.current.length >= RATE_LIMIT.maxRequests) {
      const oldestInWindow = requestTimestamps.current[0];
      const waitSeconds = Math.ceil(
        (oldestInWindow + RATE_LIMIT.windowMs - now) / 1000
      );

      setIsRateLimited(true);
      setRateLimitCountdown(waitSeconds);

      // Start countdown
      if (countdownRef.current) clearInterval(countdownRef.current);
      countdownRef.current = setInterval(() => {
        setRateLimitCountdown((prev) => {
          if (prev <= 1) {
            setIsRateLimited(false);
            if (countdownRef.current) clearInterval(countdownRef.current);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return true;
    }

    return false;
  }, []);

  /**
   * Generate map from user input
   */
  const generateMap = useCallback(
    async (formData: ProfileFormData): Promise<void> => {
      // Check rate limit
      if (checkRateLimit()) {
        throw new Error('Rate limit exceeded');
      }

      setIsLoading(true);
      setError(null);
      setResult(null);

      // Record request
      requestTimestamps.current.push(Date.now());

      try {
        // Build request payload - matches API contract exactly
        const payload = {
          profile: {
            USER_TYPE: formData.userType,
            ANNUALINCOME: formData.annualIncome,
            AGE_GROUP: formData.ageGroup,
            OCCUPATION: formData.occupation,
            PRODUCTGOAL:
              formData.productGoals.length > 0
                ? formData.productGoals
                : ['BUDGETING'],
          },
          prompt: formData.goalDescription,
        };

        // Single API call - backend handles everything else
        const response = await fetch(`${API_BASE_URL}/api/generate-map`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const errorData: ApiError = await response.json().catch(() => ({
            detail: `Request failed (${response.status})`,
          }));
          throw new Error(errorData.detail);
        }

        const data: GenerateMapResponse = await response.json();
        setResult(data);
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "We couldn't generate your map. Please try again.";
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [checkRateLimit]
  );

  /**
   * Reset state for new generation
   */
  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
    generateMap,
    result,
    error,
    isLoading,
    isRateLimited,
    rateLimitCountdown,
    reset,
  };
}
