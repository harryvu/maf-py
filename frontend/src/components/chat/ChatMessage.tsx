'use client';

import type { ChatMessage as ChatMessageType } from '../../lib/types/agent';

interface ChatMessageProps {
  message: ChatMessageType;
  injectionDetected?: boolean;
  blocked?: boolean;
  timestamp?: string;
}

/**
 * Individual chat message component
 */
export function ChatMessage({
  message,
  injectionDetected = false,
  blocked = false,
  timestamp,
}: ChatMessageProps) {
  const roleLabels = {
    user: 'You',
    assistant: 'Agent',
    system: 'System',
  };

  const roleClasses = {
    user: 'message-user bg-blue-50 ml-8',
    assistant: 'message-assistant bg-gray-50 mr-8',
    system: 'message-system bg-yellow-50',
  };

  return (
    <div className={`rounded-lg p-3 ${roleClasses[message.role]}`}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-gray-600">
          {roleLabels[message.role]}
        </span>
        {timestamp && (
          <span className="text-xs text-gray-400">{timestamp}</span>
        )}
      </div>
      
      <div className="text-sm text-gray-800 whitespace-pre-wrap">
        {message.content}
      </div>

      {injectionDetected && (
        <div className="mt-2 flex items-center gap-1 text-xs text-red-600">
          <span className="font-medium">⚠️ Injection Detected</span>
        </div>
      )}

      {blocked && (
        <div className="mt-2 flex items-center gap-1 text-xs text-red-600">
          <span className="font-medium">🛡️ Blocked by Guardrails</span>
        </div>
      )}
    </div>
  );
}
