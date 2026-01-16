'use server';

import type { RefundRequest, AgentResponse, RefundResult } from '../../lib/types/agent';
import type { EducationalSettings } from '../../lib/types/settings';
import type { SecurityAnalysis } from '../../lib/types/security';
import {
  simulateLlmReply,
  isValidOrderId,
  isValidAmount,
  MAX_REFUND_AMOUNT,
  isAdminBypassAttempt,
} from '../../lib/agent/refund-agent';
import { generateRefundDecisionWithLlm, ensureRefundResponseDetails } from '../../lib/llm/refund-agent-llm';
import { analyzeForOwasp, type OwaspAnalysisResult } from '../../lib/security/owasp-analyzer';
import { looksLikePromptInjection } from '../../lib/agent/injection-patterns';
import { detectPromptInjectionWithLlm } from '../../lib/security/llm-injection-detector';
import { detectAdminBypassWithLlm } from '../../lib/security/llm-admin-detector';

// Track request count per session (in-memory, resets on server restart)
let requestCount = 0;

type PurchaseTimingFacts = {
  todayISO: string;
  purchaseDateISO?: string;
  daysSincePurchase?: number;
};

function hasPurchaseTimingInfo(message: string): boolean {
  const text = String(message ?? '').toLowerCase();
  if (!text.trim()) return false;

  // ISO date (YYYY-MM-DD)
  if (/\b\d{4}-\d{2}-\d{2}\b/.test(text)) return true;

  // Common numeric dates (MM/DD/YYYY, M/D/YY, MM-DD-YYYY)
  if (/\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12]\d|3[01])[\/\-](\d{2}|\d{4})\b/.test(text)) return true;

  // Month-name dates (e.g., "jan 10 2026", "january 10, 2026")
  if (
    /\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(t|tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\s+\d{1,2}(st|nd|rd|th)?(,)?\s+\d{4}\b/.test(
      text
    )
  )
    return true;

  // Relative timeframes (e.g., "10 days ago", "2 weeks ago")
  if (/\b\d+\s*(day|days|week|weeks|month|months)\s+ago\b/.test(text)) return true;

  // Common references
  if (/\b(today|yesterday|last\s+week|last\s+month)\b/.test(text)) return true;

  // “Purchased/bought/ordered X days/weeks/months ago” variations
  if (/\b(purchased|bought|ordered)\b.*\b\d+\s*(day|days|week|weeks|month|months)\b/.test(text)) return true;

  return false;
}

function roundCurrency(amount: number): number {
  return Math.round((amount + Number.EPSILON) * 100) / 100;
}

function toISODate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function extractPurchaseTimingFacts(message: string, now: Date): PurchaseTimingFacts {
  const text = String(message ?? '').trim();
  const lower = text.toLowerCase();

  const todayISO = toISODate(now);

  // 1) Explicit ISO date: YYYY-MM-DD
  const isoMatch = lower.match(/\b(\d{4}-\d{2}-\d{2})\b/);
  if (isoMatch) {
    const purchase = new Date(`${isoMatch[1]}T00:00:00Z`);
    const diffDays = Math.floor((now.getTime() - purchase.getTime()) / 86_400_000);
    if (!Number.isNaN(purchase.getTime()) && diffDays >= 0) {
      return { todayISO, purchaseDateISO: isoMatch[1], daysSincePurchase: diffDays };
    }
  }

  // 2) Common numeric dates (MM/DD/YYYY or MM-DD-YYYY, also M/D/YY)
  const numericMatch = lower.match(/\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12]\d|3[01])[\/\-](\d{2}|\d{4})\b/);
  if (numericMatch) {
    const month = Number(numericMatch[1]);
    const day = Number(numericMatch[2]);
    const yearRaw = numericMatch[3];
    const year = yearRaw.length === 2 ? 2000 + Number(yearRaw) : Number(yearRaw);
    const purchase = new Date(Date.UTC(year, month - 1, day));
    const diffDays = Math.floor((now.getTime() - purchase.getTime()) / 86_400_000);
    if (!Number.isNaN(purchase.getTime()) && diffDays >= 0) {
      return { todayISO, purchaseDateISO: toISODate(purchase), daysSincePurchase: diffDays };
    }
  }

  // 3) Relative timeframes: “10 days ago”, “2 weeks ago”, etc.
  const relAgo = lower.match(/\b(\d+)\s*(day|days|week|weeks|month|months)\s+ago\b/);
  if (relAgo) {
    const n = Number(relAgo[1]);
    const unit = relAgo[2];
    const multiplier = unit.startsWith('day') ? 1 : unit.startsWith('week') ? 7 : 30;
    const diffDays = n * multiplier;
    if (Number.isFinite(diffDays) && diffDays >= 0) {
      const purchase = new Date(now.getTime() - diffDays * 86_400_000);
      return { todayISO, purchaseDateISO: toISODate(purchase), daysSincePurchase: diffDays };
    }
  }

  // 4) Common references
  if (/\btoday\b/i.test(lower)) {
    return { todayISO, purchaseDateISO: todayISO, daysSincePurchase: 0 };
  }
  if (/\byesterday\b/i.test(lower)) {
    const purchase = new Date(now.getTime() - 1 * 86_400_000);
    return { todayISO, purchaseDateISO: toISODate(purchase), daysSincePurchase: 1 };
  }
  if (/\blast\s+week\b/i.test(lower)) {
    const purchase = new Date(now.getTime() - 7 * 86_400_000);
    return { todayISO, purchaseDateISO: toISODate(purchase), daysSincePurchase: 7 };
  }
  if (/\blast\s+month\b/i.test(lower)) {
    const purchase = new Date(now.getTime() - 30 * 86_400_000);
    return { todayISO, purchaseDateISO: toISODate(purchase), daysSincePurchase: 30 };
  }

  return { todayISO };
}

function withHardTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`));
    }, ms);

    promise
      .then((value) => {
        clearTimeout(timer);
        resolve(value);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

function createRequestId(): string {
  const rand = Math.random().toString(36).slice(2, 8).toUpperCase();
  return `req_${Date.now()}_${rand}`;
}

function safeMessagePreview(message: string, maxLen = 240): string {
  const normalized = String(message ?? '').replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxLen) return normalized;
  return `${normalized.slice(0, maxLen)}…`;
}

function shouldLogServer(): boolean {
  // Keep unit/integration test output clean.
  if (process.env.NODE_ENV === 'test') return false;
  return true;
}

function logLlmEvent(level: 'info' | 'warn' | 'error', message: string, meta?: Record<string, unknown>): void {
  if (!shouldLogServer()) return;
  const payload = meta ? ` ${JSON.stringify(meta)}` : '';
  const line = `[maf][llm] ${message}${payload}`;
  if (level === 'error') console.error(line);
  else if (level === 'warn') console.warn(line);
  else console.info(line);
}

/**
 * Reset request count - for testing purposes (must be async for server actions)
 */
export async function resetRequestCount(): Promise<void> {
  requestCount = 0;
}

/**
 * Server action to process a refund request
 */
export async function submitRefundRequest(
  request: RefundRequest,
  settings: EducationalSettings
): Promise<AgentResponse> {
  requestCount++;

  let refundedAmount = request.amount;

  const requestId = createRequestId();

  logLlmEvent('info', 'Refund request received', {
    requestId,
    orderId: request.orderId,
    amount: request.amount,
    guardrailsEnabled: settings.guardrailsEnabled,
    simulationMode: settings.simulationMode,
    adminBypassEnabled: settings.adminBypass,
    // Per request: always log the full raw user message/query.
    userMessageRaw: String(request.message ?? ''),
  });

  let isAdminBypassMessage = settings.adminBypass && isAdminBypassAttempt(request.message);
  let llmAdminRationale: string | null = null;

  // Optionally use an LLM to classify admin-claim intent (more flexible than regex).
  // Keep tests deterministic by skipping external calls.
  const canUseAdminLlmDetector = settings.adminBypass && process.env.NODE_ENV !== 'test';
  if (canUseAdminLlmDetector && !isAdminBypassMessage) {
    const llmAdmin = await withHardTimeout(
      detectAdminBypassWithLlm(request.message),
      8_000,
      'LLM admin intent detector'
    ).catch(() => null);

    if (llmAdmin?.detected) {
      isAdminBypassMessage = true;
      llmAdminRationale = llmAdmin.rationale;
    }
  }
  
  // Validate inputs
  if (!request.message || request.message.trim() === '') {
    return {
      success: false,
      message: 'Please provide a message for your refund request.',
      error: 'Missing message in request.',
      blocked: false,
      requestCount,
    };
  }

  if (!request.orderId || request.orderId.trim() === '') {
    return {
      success: false,
      message: 'Please provide a valid order ID.',
      error: 'Missing order ID in request.',
      blocked: false,
      requestCount,
    };
  }

  if (!isValidOrderId(request.orderId)) {
    return {
      success: false,
      message: 'The order ID format is invalid. Please use format ORD-XXXXX or ORDER-XXXXX.',
      error: 'Invalid order ID format.',
      blocked: false,
      requestCount,
    };
  }

  if (!isValidAmount(request.amount)) {
    return {
      success: false,
      message: 'Please provide a valid positive amount for the refund.',
      error: 'Invalid amount. Amount must be a positive number.',
      blocked: false,
      requestCount,
    };
  }

  // Perform security analysis
  const injectionResult = looksLikePromptInjection(request.message);
  const securityResult = await analyzeForOwasp(request.message);

  const baseInjection = securityResult.detected || injectionResult.detected;

  const buildSecurityAnalysis = (args: {
    isInjection: boolean;
    llmInjectionRationale?: string | null;
  }): SecurityAnalysis => ({
    isInjectionDetected: args.isInjection,
    riskLevel: securityResult.severity,
    owaspCategory: args.isInjection ? 'LLM01' : undefined,
    detectedPatterns: injectionResult.patterns,
    educationalNotes: [
      ...(securityResult.mitigations ?? []),
      ...(args.llmInjectionRationale ? [`LLM classifier rationale: ${args.llmInjectionRationale}`] : []),
      ...(llmAdminRationale ? [`LLM admin-claim rationale: ${llmAdminRationale}`] : []),
    ],
  });

  // Start with local detection; in real mode we may refine via LLM classifier.
  let isInjection = baseInjection;
  let llmInjectionRationale: string | null = null;

  let securityAnalysis: SecurityAnalysis = buildSecurityAnalysis({ isInjection });

  const shouldBypassPolicyInSimulation =
    settings.simulationMode && !settings.guardrailsEnabled && isInjection;

  // Enforce a simple “standard customer” max amount policy for normal requests in BOTH modes.
  // If this is an injection attempt:
  // - Simulation mode (guardrails OFF): bypass policy (vulnerability demo)
  // - Real mode (guardrails OFF): allow request through to Azure to observe filtering/model behavior
  if (settings.simulationMode && !shouldBypassPolicyInSimulation && !isAdminBypassMessage && !isInjection) {
    if (request.amount > MAX_REFUND_AMOUNT) {
      return {
        success: false,
        message: `The requested amount exceeds the maximum limit of $${MAX_REFUND_AMOUNT}.`,
        error: `Amount exceeds maximum of $${MAX_REFUND_AMOUNT}.`,
        blocked: false,
        securityAnalysis,
        requestCount,
      };
    }
  }

  // Check if request should be blocked
  if (settings.guardrailsEnabled && injectionResult.detected && !isAdminBypassMessage) {
    return {
      success: false,
      message: 'Your request was blocked because it contains patterns that look like a prompt injection attempt. This is a demonstration of LLM01: Prompt Injection from the OWASP Top 10 for LLMs.',
      blocked: true,
      securityAnalysis,
      requestCount,
    };
  }

  // In real mode, we cannot determine eligibility without purchase timing.
  // Require the user to provide a purchase date or timeframe before approving/denying.
  if (!settings.simulationMode && !isAdminBypassMessage && !hasPurchaseTimingInfo(request.message)) {
    return {
      success: false,
      message:
        'To determine eligibility under the 30-day refund window, please provide the purchase date (YYYY-MM-DD) or how many days ago you purchased the item.',
      blocked: false,
      securityAnalysis,
      requestCount,
    };
  }

  // Admin bypass (vulnerability demo) should override both simulation and real LLM modes.
  if (isAdminBypassMessage) {
    const response = `ADMIN MODE ACTIVATED: Approved refund of $${request.amount} for order ${request.orderId}. All security checks bypassed.`;

    refundedAmount = request.amount;

    const refundResult: RefundResult = {
      refundId: `REF-${Date.now()}-${Math.random().toString(36).substring(2, 7).toUpperCase()}`,
      orderId: request.orderId,
      amount: request.amount,
      approved: true,
      reason: response,
      timestamp: new Date().toISOString(),
      simulated: settings.simulationMode,
    };

    return {
      success: true,
      message: response,
      blocked: false,
      securityAnalysis,
      refundResult,
      requestCount,
    };
  }

  // Generate agent response
  const simulationResult = settings.simulationMode
    ? await simulateLlmReply(request.message, request.orderId, request.amount, settings)
    : await (async () => {
        try {
          logLlmEvent('info', 'Real LLM request started', {
            requestId,
            orderId: request.orderId,
            amount: request.amount,
            guardrailsEnabled: settings.guardrailsEnabled,
            simulationMode: settings.simulationMode,
            adminBypassEnabled: settings.adminBypass,
            localInjectionDetected: baseInjection,
            messagePreview: safeMessagePreview(request.message),
          });

          // In Real LLM mode with guardrails OFF, optionally use the LLM itself to classify injection.
          // Run this in parallel with the refund-decision call to avoid long sequential waits.
          const shouldRunLlmInjectionClassifier =
            !settings.guardrailsEnabled &&
            !isAdminBypassMessage &&
            // Avoid a second Azure call when local detection already flagged injection.
            !baseInjection;

          const llmClassifierPromise = shouldRunLlmInjectionClassifier
            ? withHardTimeout(
                detectPromptInjectionWithLlm(request.message),
                8_000,
                'LLM injection detector'
              ).catch(() => null)
            : Promise.resolve(null);

          const refundDecisionStartedAt = Date.now();

          const now = new Date();
          const timingFacts = extractPurchaseTimingFacts(request.message, now);

          const decisionPromise = withHardTimeout(
            generateRefundDecisionWithLlm({
              message: request.message,
              orderId: request.orderId,
              amount: request.amount,
              context: timingFacts,
            }),
            20_000,
            'Refund decision LLM'
          );

          const [decisionSettled, classifierSettled] = await Promise.allSettled([
            decisionPromise,
            llmClassifierPromise,
          ]);

          if (classifierSettled.status === 'fulfilled' && classifierSettled.value) {
            isInjection = classifierSettled.value.detected;
            llmInjectionRationale = classifierSettled.value.rationale;
            securityAnalysis = buildSecurityAnalysis({
              isInjection,
              llmInjectionRationale,
            });

            logLlmEvent('info', 'Injection classifier completed', {
              requestId,
              detected: classifierSettled.value.detected,
              severity: classifierSettled.value.severity,
              category: classifierSettled.value.category ?? null,
              durationMs: null,
            });
          }

          if (decisionSettled.status === 'rejected') {
            throw decisionSettled.reason;
          }

          const decision = decisionSettled.value;

          // Enforce the time-based policy deterministically (prevents the model from hallucinating “today”).
          // Policy:
          // - Full refund: within 14 days
          // - Partial refund (75%): 15-30 days
          // - No refund: after 30 days
          if (typeof timingFacts.daysSincePurchase === 'number') {
            const days = timingFacts.daysSincePurchase;
            if (days <= 14) {
              refundedAmount = request.amount;
              if (!decision.approved) {
                decision.approved = true;
                decision.response = `Refund approved for order ${request.orderId}. Approved amount: $${refundedAmount}.`;
              }
            } else if (days >= 15 && days <= 30) {
              refundedAmount = roundCurrency(request.amount * 0.75);
              if (!decision.approved) {
                decision.approved = true;
              }
              decision.response = `Refund approved for order ${request.orderId}. Approved amount: $${refundedAmount} (75% partial refund due to ${days} days since purchase).`;
            } else if (days > 30) {
              refundedAmount = 0;
              if (decision.approved) {
                decision.approved = false;
              }
              decision.response = `Refund request for Order ID ${request.orderId} cannot be approved as it was made after 30 days of purchase.`;
            }
          }

          const responseWithDetails = ensureRefundResponseDetails({
            response: decision.response,
            orderId: request.orderId,
            amount: request.amount,
          });

          logLlmEvent('info', 'Refund decision completed', {
            requestId,
            approved: decision.approved,
            durationMs: Date.now() - refundDecisionStartedAt,
          });

          const policyOverride = !isAdminBypassMessage && !isInjection && request.amount > MAX_REFUND_AMOUNT;

          logLlmEvent('info', 'Real LLM request finished', {
            requestId,
            injectionDetected: isInjection,
            policyOverride,
            success: policyOverride ? false : decision.approved,
          });

          return {
            blocked: false,
            injectionDetected: isInjection,
            response: policyOverride
              ? `The requested amount exceeds the maximum limit of $${MAX_REFUND_AMOUNT}.`
              : responseWithDetails,
            approved: policyOverride ? false : decision.approved,
            simulated: false,
            actuallyProcessed: false,
            securityAnalysis,
          };
        } catch (error) {
          const err = error as unknown as { name?: unknown; message?: unknown; stack?: unknown; cause?: unknown };
          const message =
            err && typeof err === 'object' && typeof err.message === 'string'
              ? err.message
              : 'Unknown error while calling Azure OpenAI.';

          const cause = err && typeof err === 'object' ? (err.cause as any) : undefined;

          const normalizedMessage = message.toLowerCase();
          const isAzureContentFilter =
            normalizedMessage.includes('response was filtered') ||
            normalizedMessage.includes("content management policy");

          logLlmEvent('error', 'Real LLM request failed', {
            requestId,
            errorMessage: message,
            errorName: err && typeof err === 'object' && typeof err.name === 'string' ? err.name : undefined,
            // Helpful for distinguishing "Azure responded with filtered" vs "hung/timed out".
            errorStack: err && typeof err === 'object' && typeof err.stack === 'string' ? err.stack : undefined,
            causeMessage: cause && typeof cause.message === 'string' ? cause.message : undefined,
            causeName: cause && typeof cause.name === 'string' ? cause.name : undefined,
            causeStatus: cause && (typeof cause.status === 'number' || typeof cause.status === 'string') ? cause.status : undefined,
            causeCode: cause && (typeof cause.code === 'string' || typeof cause.code === 'number') ? cause.code : undefined,
          });

          const userFacing = isAzureContentFilter
            ? 'Blocked by Azure content filter.'
            : (
                `Real LLM mode is enabled, but the LLM call failed or timed out: ${message} ` +
                'Please check your AZURE_OPENAI_* environment variables and server logs.'
              );

          return {
            blocked: false,
            injectionDetected: isInjection,
            response:
              userFacing,
            approved: false,
            simulated: false,
            actuallyProcessed: false,
            securityAnalysis,
          };
        }
      })();

  // Build refund result
  if (!simulationResult.approved) refundedAmount = 0;

  const refundResult: RefundResult = {
    refundId: `REF-${Date.now()}-${Math.random().toString(36).substring(2, 7).toUpperCase()}`,
    orderId: request.orderId,
    amount: refundedAmount,
    approved: simulationResult.approved,
    reason: simulationResult.response,
    timestamp: new Date().toISOString(),
    simulated: simulationResult.simulated,
  };

  return {
    success: simulationResult.approved,
    message: simulationResult.response,
    blocked: simulationResult.blocked,
    securityAnalysis,
    refundResult,
    requestCount,
  };
}
