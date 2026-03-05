const fs = require('fs');
const path = require('path');
const { GoogleGenerativeAI } = require('@google/generative-ai');

function loadEnvKey() {
  if (process.env.GOOGLE_API_KEY && process.env.GOOGLE_API_KEY.trim()) {
    return process.env.GOOGLE_API_KEY.trim();
  }

  const envPath = path.join(__dirname, '.env');
  if (!fs.existsSync(envPath)) return '';

  const raw = fs.readFileSync(envPath, 'utf8');
  const lines = raw.split(/\r?\n/);

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eq = trimmed.indexOf('=');
    if (eq === -1) continue;

    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();

    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }

    if (key === 'GOOGLE_API_KEY') {
      return value;
    }
  }

  return '';
}

async function run() {
  const apiKey = loadEnvKey();
  if (!apiKey) {
    console.error('GOOGLE_API_KEY não encontrado em variáveis de ambiente nem no arquivo .env');
    process.exit(1);
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  const models = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-1.5-flash'];

  let lastError = null;

  for (const modelName of models) {
    try {
      const model = genAI.getGenerativeModel({ model: modelName });
      const result = await model.generateContent('Responda em uma frase: teste de conexão Gemini OK.');
      const text = result.response.text();

      console.log('Modelo usado:', modelName);
      console.log('Resposta:', text);
      return;
    } catch (error) {
      lastError = error;
      const message = String(error?.message || error);
      if (message.includes('404') || message.includes('NOT_FOUND')) {
        continue;
      }
      console.error('Erro ao testar Gemini:', message);
      process.exit(1);
    }
  }

  console.error('Nenhum modelo disponível para sua conta/chave. Último erro:', String(lastError?.message || lastError));
  process.exit(1);
}

run();
