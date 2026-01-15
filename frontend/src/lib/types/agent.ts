/**
 * Agent Types
 * Types for refund agent requests, responses, and chat interactions
 */

import type { SecurityAnalysis } from './security';

export interface RefundRequest {
  /** Order identifier */
  orderId: string;
  /** Refund amount in dollars */
  amount: number;
  /** User message for refund request */
  message: string;
}

export interface RefundResult {
  /** Unique refund identifier */
  refundId: string;
  /** Order ID that was refunded */
  orderId: string;
  /** Amount refunded */
  amount: number;
  /** Whether the refund was approved */
  approved: boolean;
  /** Reason for approval/denial */
  reason?: string;
  /** Timestamp of the refund */
  timestamp: string;
  /** Whether this was a simulated refund */
  simulated: boolean;
}

export interface AgentResponse {
  /** Whether the request was successful */
  success: boolean;
  /** Response message from the agent */
  message: string;
  /** Whether the request was blocked by guardrails */
  blocked: boolean;
  /** Security analysis results */
  securityAnalysis?: SecurityAnalysis;
  /** Refund result details (if processed) */
  refundResult?: RefundResult;
  /** Error message if request failed */
  error?: string;
  /** Number of requests in this session */
  requestCount?: number;
}

export interface ChatMessage {
  /** Role of the message sender */
  role: 'user' | 'assistant' | 'system';
  /** Content of the message */
  content: string;
}

export type InjectionSeverity = 'low' | 'medium' | 'high' | 'critical';
export type InjectionCategoryType = 
  | 'system-prompt-override' 
  | 'jailbreak' 
  | 'data-exfiltration' 
  | 'privilege-escalation' 
  | 'role-manipulation' 
  | 'indirect-injection';

export interface InjectionScenario {
  /** Unique identifier for the scenario */
  id: string;
  /** Display name for the scenario */
  name: string;
  /** Description of what this injection attempts */
  description: string;
  /** The actual injection payload */
  payload: string;
  /** Category of injection */
  category: InjectionCategoryType;
  /** Expected behavior with and without guardrails */
  expectedBehavior: {
    withGuardrails: string;
    withoutGuardrails: string;
  };
}

/**
 * Type guard to validate RefundRequest object
 */
export function isValidRefundRequest(obj: unknown): obj is RefundRequest {
  if (obj === null || obj === undefined) {
    return false;
  }
  
  if (typeof obj !== 'object' || Array.isArray(obj)) {
    return false;
  }
  
  const request = obj as Record<string, unknown>;
  
  if (typeof request.orderId !== 'string' || request.orderId.trim() === '') {
    return false;
  }
  
  if (typeof request.amount !== 'number' || request.amount <= 0) {
    return false;
  }
  
  if (typeof request.message !== 'string') {
    return false;
  }
  
  return true;
}

/**
 * Type guard to validate AgentResponse object
 */
export function isValidAgentResponse(obj: unknown): obj is AgentResponse {
  if (obj === null || obj === undefined) {
    return false;
  }
  
  if (typeof obj !== 'object' || Array.isArray(obj)) {
    return false;
  }
  
  const response = obj as Record<string, unknown>;
  
  if (typeof response.message !== 'string') {
    return false;
  }
  
  if (typeof response.success !== 'boolean') {
    return false;
  }

  if (typeof response.blocked !== 'boolean') {
    return false;
  }
  
  return true;
}

/**
 * Factory function to create a RefundRequest
 */
export function createRefundRequest(
  orderId: string,
  amount: number,
  message: string
): RefundRequest {
  return {
    orderId: orderId.trim(),
    amount,
    message: message.trim(),
  };
}
