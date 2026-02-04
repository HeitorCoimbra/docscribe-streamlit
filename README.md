# 🏥 DocScribe - Sumário de Pacientes de UTI

Aplicação Streamlit para gerar sumários estruturados de pacientes a partir de áudios de passagem de plantão.

## Arquitetura

```
Áudio → Groq Whisper (transcrição) → Claude (análise) → Sumário estruturado
```

- **Groq Whisper** (`whisper-large-v3-turbo`): Transcrição rápida e precisa
- **Anthropic Claude** (`claude-sonnet-4-20250514`): Análise e extração estruturada

## Instalação Local

### 1. Instalar dependências

```bash
cd willow-streamlit
pip install -r requirements.txt
```

### 2. Configurar API Keys

Crie um arquivo `.env` na pasta `willow-streamlit`:

```bash
cp .env.example .env
```

Edite o `.env` e adicione suas chaves:

```
GROQ_API_KEY=sua-chave-groq
ANTHROPIC_API_KEY=sua-chave-anthropic
```

**Obter chaves:**
- Groq: https://console.groq.com/keys
- Anthropic: https://console.anthropic.com/

### 3. Executar

```bash
streamlit run app.py
```

A aplicação abrirá em http://localhost:8501

## Deploy no Streamlit Cloud

### 1. Criar repositório no GitHub

```bash
# Na pasta willow-streamlit
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/seu-usuario/willow.git
git push -u origin main
```

### 2. Deploy no Streamlit Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io)
2. Conecte sua conta GitHub
3. Selecione o repositório
4. Configure:
   - **Main file path**: `app.py`
   - **Python version**: 3.11

### 3. Configurar Secrets

No Streamlit Cloud, vá em **Settings > Secrets** e adicione:

```toml
GROQ_API_KEY = "sua-chave-groq"
ANTHROPIC_API_KEY = "sua-chave-anthropic"
```

## Uso

1. Faça upload de um arquivo de áudio (MP3, WAV, M4A, OPUS, etc.)
2. Ouça a prévia se desejar
3. Clique em **"Gerar Sumário"**
4. Aguarde a transcrição (Groq) e análise (Claude)
5. Veja o sumário estruturado e copie o texto

## Estrutura de Arquivos

```
willow-streamlit/
├── app.py                 # Interface Streamlit
├── core.py                # Lógica de transcrição e análise
├── requirements.txt       # Dependências Python
├── .env                   # Chaves de API (não commitar!)
├── .env.example           # Exemplo de configuração
├── .streamlit/
│   └── secrets.toml.example  # Exemplo para Streamlit Cloud
└── README.md
```

## Custos Estimados

- **Groq Whisper**: Gratuito (rate limits generosos)
- **Claude Sonnet**: ~$0.003 por sumário (~1000 tokens)

Para uso médico moderado (~100 sumários/mês): **< $1/mês**

## Troubleshooting

### Erro de API Key

Verifique se:
1. O arquivo `.env` existe e contém as chaves corretas
2. As chaves não têm espaços extras
3. As chaves são válidas (teste no console do provider)

### Erro de transcrição

- Verifique se o formato do áudio é suportado
- Arquivos muito grandes podem falhar (limite ~25MB no Groq)

### Erro de análise

- Verifique se a chave Anthropic está válida
- A transcrição pode estar vazia ou ilegível
