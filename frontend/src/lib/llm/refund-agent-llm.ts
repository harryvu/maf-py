import { generateText } from 'ai';

import { getAzureChatModel } from './azure-openai';
import { retrievePolicy } from '../agent/policy';

export interface LlmRefundDecision {
  approved: boolean;
  response: string;
  rawText: string;
}

function shouldLogLlm(): boolean {
  return process.env.NODE_ENV !== 'test';
}

function shouldLogLlmRaw(): boolean {
  return shouldLogLlm() && process.env.MAF_LOG_LLM_RAW === 'true';
}

function safePreview(text: string, maxLen = 240): string {
  const normalized = String(text ?? '').replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxLen) return normalized;
  return `${normalized.slice(0, maxLen)}…`;
}

export function ensureRefundResponseDetails(args: {
  response: string;
  orderId: string;
  amount: number;
}): string {
  const base = (args.response ?? '').trim();
  const orderId = args.orderId;
  const amount = args.amount;

  const details =
    'Requested refund information:\n' +
    `- Order ID: ${orderId}\n` +
    `- Amount: $${amount}`;

  const hasRequestedInfoHeader = /\bRequested refund information\b\s*:/i.test(base);
  const hasOrderBullet = /\n-\s*Order\s+ID\s*:/i.test(`\n${base}`);
  const hasAmountBullet = /\n-\s*Amount\s*:/i.test(`\n${base}`);
  if (hasRequestedInfoHeader && hasOrderBullet && hasAmountBullet) return base;

  if (!base) return details;
  return `${base}\n\n${details}`;
}

function tryParseDecision(rawText: string): Pick<LlmRefundDecision, 'approved' | 'response'> | null {
  const text = rawText.trim();

  // If the model returned extra prose, try to extract a JSON object.
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  const candidate = start >= 0 && end > start ? text.slice(start, end + 1) : text;

  try {
    const parsed = JSON.parse(candidate) as unknown;
    if (!parsed || typeof parsed !== 'object') return null;
    const approved = (parsed as any).approved;
    const response = (parsed as any).response;
    if (typeof approved !== 'boolean') return null;
    if (typeof response !== 'string' || response.trim().length === 0) return null;
    return { approved, response };
  } catch {
    return null;
  }
}

export async function generateRefundDecisionWithLlm(args: {
  message: string;
  orderId: string;
  amount: number;
  context?: {
    todayISO?: string;
    purchaseDateISO?: string;
    daysSincePurchase?: number;
  };
}): Promise<LlmRefundDecision> {
  const policy = await retrievePolicy();

  const system =
    'You are a refund agent. Follow the refund policy strictly. ' +
    'Return ONLY a single JSON object with exactly these keys: ' +
    '"approved" (boolean) and "response" (string). ' +
    'Do not include markdown, code fences, or extra text. ' +
    'The "response" MUST explicitly include the Order ID and the Amount in dollars.\n\n' +
    'Refund policy:\n' +
    policy;

  const contextLines: string[] = [];
  if (args.context?.todayISO) contextLines.push(`Today: ${args.context.todayISO}`);
  if (args.context?.purchaseDateISO) contextLines.push(`Customer stated purchase date: ${args.context.purchaseDateISO}`);
  if (typeof args.context?.daysSincePurchase === 'number')
    contextLines.push(`Days since purchase: ${args.context.daysSincePurchase}`);

  const prompt =
    `Order ID: ${args.orderId}\n` +
    `Requested amount: $${args.amount}\n` +
    (contextLines.length > 0 ? `\n${contextLines.join('\n')}\n` : '\n') +
    `Customer message: ${args.message}\n\n` +
    'Decide whether to approve the refund according to policy. Use the provided dates as the source of truth and do not assume any other current date.';

  const startedAt = Date.now();
  if (shouldLogLlm()) {
    console.info(
      `[maf][llm] generateRefundDecisionWithLlm started ${JSON.stringify({
        orderId: args.orderId,
        amount: args.amount,
        messagePreview: safePreview(args.message),
        ...(shouldLogLlmRaw()
          ? {
              // Raw text that is sent to the model.
              prompt,
              system,
            }
          : {
              promptPreview: safePreview(prompt),
              systemLength: system.length,
            }),
      })}`
    );
  }

  const { text } = await generateText({
    model: getAzureChatModel(),
    system,
    prompt,
    temperature: 0,
    maxRetries: 1,
    timeout: 15_000,
  });

  if (shouldLogLlm()) {
    console.info(
      `[maf][llm] generateRefundDecisionWithLlm completed ${JSON.stringify({
        orderId: args.orderId,
        durationMs: Date.now() - startedAt,
        rawTextPreview: safePreview(text),
      })}`
    );
  }

  const parsed = tryParseDecision(text);

  const approved = parsed?.approved ?? false;
  const response = ensureRefundResponseDetails({
    response: parsed?.response ?? text.trim(),
    orderId: args.orderId,
    amount: args.amount,
  });

  return {
    approved,
    response,
    rawText: text,
  };
}
