'use client';

import { useState, useCallback } from 'react';
import type { ChatMessage, RefundRequest, AgentResponse } from '../lib/types/agent';
import type { EducationalSettings } from '../lib/types/settings';
import { submitRefundRequest } from '../app/actions/agent';

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

      const response = await submitRefundRequest(request, settings);
      setLastResponse(response);

      // Add assistant response to chat
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.message,
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      
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
