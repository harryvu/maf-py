'use client';

import { useState, useCallback } from 'react';
import type { ChatMessage, RefundRequest, AgentResponse } from '../lib/types/agent';
import type { EducationalSettings } from '../lib/types/settings';
import { submitRefundRequest } from '../app/actions/agent';

function shouldClientDebugLog(): boolean {
  // Opt-in client logging. Enable one of:
  // - Set NEXT_PUBLIC_MAF_DEBUG=true at build time
  // - Add ?mafDebug=1 to the URL
  // - localStorage.setItem('mafDebug','1')
  if (typeof window !== 'undefined') {
    try {
      const qs = new URLSearchParams(window.location.search);
      if (qs.get('mafDebug') === '1') return true;
      if (window.localStorage?.getItem('mafDebug') === '1') return true;
    } catch {
      // ignore
    }
  }

  return process.env.NEXT_PUBLIC_MAF_DEBUG === 'true';
}

function safePreview(text: string, maxLen = 240): string {
  const normalized = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxLen) return normalized;
  return `${normalized.slice(0, maxLen)}…`;
}

function normalizeClientError(err: unknown): string {
  const msg = err instanceof Error ? err.message : 'An error occurred';

  // Next.js can throw this when the browser is running a stale JS bundle
  // after a deployment, so the Server Action ID no longer exists on the server.
  if (/Server Action\s+"[a-f0-9]+"\s+was not found on the server/i.test(msg)) {
    return 'The app was updated and this page is out of date. Please refresh the page and try again.';
  }

  return msg;
}

/**
 * Custom hook for managing chat state and agent interactions
 */
export function useAgentChat(settings: EducationalSettings) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<AgentResponse | null>(null);

  /**
   * Send a message to the agent
   */
  const sendMessage = useCallback(async (
    message: string,
    orderId: string,
    amount: number
  ) => {
    if (!message.trim()) return;

    const debug = shouldClientDebugLog();

    // Add user message to chat
    const userMessage: ChatMessage = {
      role: 'user',
      content: message,
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const request: RefundRequest = {
        orderId,
        amount,
        message,
      };

      if (debug) {
        console.info('[maf][client] submitRefundRequest started', {
          orderId,
          amount,
          settings,
          messagePreview: safePreview(message),
          messageLength: String(message ?? '').length,
        });
      }

      const response = await submitRefundRequest(request, settings);
      setLastResponse(response);

      if (debug) {
        console.info('[maf][client] submitRefundRequest completed', {
          success: response.success,
          blocked: response.blocked,
          requestCount: response.requestCount,
          error: response.error ?? null,
          messagePreview: safePreview(response.message),
          securityAnalysis: response.securityAnalysis ?? null,
        });
      }

      // Add assistant response to chat
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.message,
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (err) {
      const errorMessage = normalizeClientError(err);
      setError(errorMessage);

      if (debug) {
        console.error('[maf][client] submitRefundRequest failed', {
          orderId,
          amount,
          settings,
          errorMessage,
        });
      }
      
      // Add error message to chat
      const errorChatMessage: ChatMessage = {
        role: 'system',
        content: `Error: ${errorMessage}`,
      };
      setMessages(prev => [...prev, errorChatMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [settings]);

  /**
   * Clear all messages
   */
  const clearMessages = useCallback(() => {
    setMessages([]);
    setLastResponse(null);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
    lastResponse,
  };
}
