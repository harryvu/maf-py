This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Real LLM Mode (Azure OpenAI)

By default, the app runs in **Simulation Mode** (no external LLM calls). If you turn **Simulation Mode** OFF in the UI, the server action will call Azure OpenAI via the AI SDK.

Set these environment variables (e.g., in your shell or a `.env.local` in `frontend/`):

```bash
# Either set BASE_URL or RESOURCE_NAME
AZURE_OPENAI_BASE_URL="https://<your-resource>.openai.azure.com/openai"
# or
AZURE_OPENAI_RESOURCE_NAME="<your-resource-name>"

# Alternative (common) endpoint form (the code will normalize it to BASE_URL)
AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"

AZURE_OPENAI_API_KEY="<your-api-key>"
AZURE_OPENAI_DEPLOYMENT="<your-chat-deployment-id>"

# Alternative deployment name env var
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="<your-chat-deployment-id>"

# Optional
AZURE_OPENAI_API_VERSION="preview"
```

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
