# API Key Setup Instructions

## OpenRouter Setup (Free Alternative to OpenAI)
1. Go to [OpenRouter](https://openrouter.ai/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Add to your `.env` file:
   ```
   OPENROUTER_API_KEY=sk-or-your-key-here
   ```

## Benefits of OpenRouter:
- **Free tier available** (with usage limits)
- **Multiple models** (GPT-3.5, Claude, etc.)
- **Lower costs** than OpenAI direct
- **Same API format** as OpenAI
- **No credit card required** for free tier

## Your .env file should look like this:
```env
OPENROUTER_API_KEY=sk-or-your-openrouter-key-here
```

## Model Names for OpenRouter:
- `gpt-3.5-turbo` (cheapest, recommended)
- `gpt-4` (more capable)
- `claude-3-haiku` (Anthropic's model)
- `llama-3.1-8b-instruct` (Meta's model)
- `mistral-7b-instruct` (Mistral's model)

## Getting Started:
1. Visit [openrouter.ai](https://openrouter.ai/)
2. Click "Get Started" and create an account
3. Go to "API Keys" in your dashboard
4. Copy your API key
5. Add it to your `.env` file
6. Run your script - it will automatically use OpenRouter!

The script now exclusively uses OpenRouter for all LLM queries!
