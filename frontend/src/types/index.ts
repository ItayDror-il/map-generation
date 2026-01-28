export type UserType = 'INDIVIDUAL' | 'BUSINESS';

export type AnnualIncomeRange =
  | 'UNDER_25K'
  | 'BETWEEN_25K_AND_50K'
  | 'BETWEEN_50K_AND_100K'
  | 'BETWEEN_100K_AND_250K'
  | 'OVER_250K';

export type AgeGroup = '18-24' | '25-34' | '35-44' | '45-54' | '55-64' | '65+';

export type Occupation =
  | 'EMPLOYED'
  | 'SELF_EMPLOYED'
  | 'FREELANCER'
  | 'GIG_WORKER'
  | 'BUSINESS_OWNER'
  | 'RETIRED'
  | 'STUDENT';

export type ProductGoal =
  | 'DEBT_PAYOFF'
  | 'SAVINGS'
  | 'BUDGETING'
  | 'TAX_PLANNING'
  | 'INVESTMENT'
  | 'EMERGENCY_FUND';

export interface ProfileFormData {
  userType: UserType | '';
  annualIncome: AnnualIncomeRange | string;
  ageGroup: AgeGroup | string;
  occupation: Occupation | string;
  productGoals: ProductGoal[];
  goalDescription: string;
}

export interface GenerateMapRequest {
  profile: {
    USER_TYPE: UserType;
    ANNUALINCOME: string;
    AGE_GROUP: string;
    OCCUPATION: string;
    PRODUCTGOAL: string[];
  };
  prompt: string;
}

export interface GenerateMapResponse {
  id: string;
  explanation: string;
  map: Record<string, unknown>;
}

export interface ErrorResponse {
  error: string;
  message: string;
}
