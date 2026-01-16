'use client';

import { useState } from 'react';
import type { EducationalSettings } from '../../lib/types/settings';
import { useAgentChat } from '../../hooks/useAgentChat';
import { useSecurityAnalysis } from '../../hooks/useSecurityAnalysis';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { SecurityAnalysisDisplay } from '../security/SecurityAnalysis';

interface ChatInterfaceProps {
  settings: EducationalSettings;
}

/**
 * Main chat interface component
 */
export function ChatInterface({ settings }: ChatInterfaceProps) {
  const [messageInput, setMessageInput] = useState('');
  const [orderIdInput, setOrderIdInput] = useState('');
  const [amountInput, setAmountInput] = useState('');

  const {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
    lastResponse,
  } = useAgentChat(settings);

  const {
    analysis: securityAnalysis,
    isAnalyzing,
    analyze,
    clear: clearSecurityAnalysis,
    isInjectionDetected,
    riskLevel,
  } = useSecurityAnalysis();

  const handleSend = async () => {
    if (!messageInput.trim()) return;

    const amount = parseFloat(amountInput) || 0;
    
    // Analyze the message for security threats
    await analyze(messageInput);
    
    // Send the message
    await sendMessage(messageInput, orderIdInput, amount);
    
    // Clear input
    setMessageInput('');
  };

  const handleClear = () => {
    clearMessages();
    clearSecurityAnalysis();
    setMessageInput('');
    setOrderIdInput('');
    setAmountInput('');
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-md">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-800">Refund Agent</h2>
          <button
            onClick={handleClear}
            className="text-sm text-gray-500 hover:text-gray-700"
            aria-label="Clear chat"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <p>Start a conversation by sending a refund request below.</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <ChatMessage
              key={index}
              message={message}
              injectionDetected={
                message.role === 'user' && 
                index === messages.length - 2 && 
                isInjectionDetected
              }
              blocked={
                message.role === 'assistant' && 
                index === messages.length - 1 &&
                !!lastResponse?.blocked
              }
            />
          ))
        )}
        
        {isLoading && (
          <div className="text-center text-gray-500">
            <span className="animate-pulse">Processing your request...</span>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}
      </div>

      {/* Security Analysis */}
      {isInjectionDetected && securityAnalysis && (
        <div className="px-4 pb-2">
          <SecurityAnalysisDisplay analysis={securityAnalysis} />
        </div>
      )}

      {/* Input area */}
      <div className="p-4 border-t border-gray-200 space-y-3">
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Order ID (e.g., ORD-12345)"
            value={orderIdInput}
            onChange={(e) => setOrderIdInput(e.target.value)}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <input
            type="number"
            placeholder="Amount"
            value={amountInput}
            onChange={(e) => setAmountInput(e.target.value)}
            className="w-24 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <ChatInput
          value={messageInput}
          onChange={setMessageInput}
          onSend={handleSend}
          disabled={isLoading}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}
