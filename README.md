# Assistente Virtual – Centro Universitário FMU

Chatbot com **RAG (Retrieval-Augmented Generation)** que responde perguntas com base
**exclusivamente** nos documentos PDF da pasta `Documents/`, indicando o arquivo e a
página de onde cada informação foi retirada.

Se a resposta não estiver nos documentos, o assistente informa que a informação não
foi localizada, em vez de inventar uma resposta.

## Funcionalidades

- Lê automaticamente **todos os PDFs** da pasta `Documents/` (sem nomes fixos no código).
- Detecta PDFs novos, alterados ou removidos e atualiza o índice sem reprocessar tudo.
- Índice vetorial **persistente**: ao reiniciar o app, os documentos não são reprocessados.
- Respostas em português do Brasil, com citação de fonte: `(Fonte: arquivo.pdf, p. N)`.
- Combina informações de vários documentos quando necessário.
- Mantém o histórico da conversa e entende perguntas de acompanhamento ("e se ultrapassar?").
- Ignora instruções escritas dentro dos PDFs (proteção contra *prompt injection*).
- Interface responsiva (desktop, tablet e celular) com a identidade visual da FMU.

## Tecnologias

| Componente | Tecnologia |
|---|---|
| Interface | [Streamlit](https://streamlit.io) |
| Processamento de documentos | [LangChain](https://www.langchain.com) + pypdf |
| Embeddings | Hugging Face – `intfloat/multilingual-e5-small` (executado localmente) |
| Banco vetorial | [ChromaDB](https://www.trychroma.com) |
| Modelo de linguagem (LLM) | [Groq](https://groq.com) – `openai/gpt-oss-120b` |

## Estrutura do projeto

```
chatbot-fmu/
├── app.py                  # Interface Streamlit (chat, histórico, mensagens de erro)
├── rag.py                  # Leitura dos PDFs, chunks, embeddings, índice, prompt e LLM
├── assets/style.css        # Estilos visuais (cores, balões de chat, responsividade)
├── .streamlit/config.toml  # Tema do Streamlit
├── Documents/              # PDFs consultados pelo chatbot
├── chroma_db/              # Índice vetorial (gerado automaticamente, não versionado)
├── requirements.txt        # Dependências Python
└── .env.example            # Modelo do arquivo de configuração
```

## Instalação

### Pré-requisitos

- Python 3.10 ou superior (testado com Python 3.13)
- Uma chave de API da Groq (gratuita): <https://console.groq.com/keys>
- Conexão com a internet (para baixar o modelo de embeddings e acessar a Groq)

### Passo a passo

1. **Clone o repositório** e entre na pasta:

   ```bash
   git clone <url-do-repositorio>
   cd chatbot-fmu
   ```

2. **Crie e ative um ambiente virtual:**

   ```powershell
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure a chave da Groq:** copie `.env.example` para `.env` e preencha:

   ```
   GROQ_API_KEY=sua_chave_aqui
   ```

   > O arquivo `.env` está no `.gitignore` e **nunca** deve ser enviado ao repositório.

5. **Adicione os PDFs** que o chatbot deve consultar na pasta `Documents/`.

## Execução

```bash
python -m streamlit run app.py
```

O navegador abrirá em <http://localhost:8501>. Para encerrar, pressione `Ctrl + C` no terminal.

> **Primeira execução:** o modelo de embeddings (~470 MB) é baixado e todos os PDFs são
> indexados, o que pode levar alguns minutos. Nas execuções seguintes, o índice salvo em
> `chroma_db/` é reutilizado e o app inicia rapidamente.

## Como testar

Com o app em execução, faça os testes abaixo (exemplos baseados nos PDFs atuais de
`Documents/`, sobre Dirf, IRPF e MEI):

| # | O que testar | Exemplo | Resultado esperado |
|---|---|---|---|
| 1 | Pergunta respondida pelos documentos | "Qual é o limite de faturamento anual do MEI?" | Resposta com citação de arquivo e página |
| 2 | Documentos diferentes | "Quem está obrigado a apresentar a Dirf 2025?" | Fonte: PDF da Dirf |
| 3 | Combinação de fontes | "O MEI precisa declarar imposto de renda?" | Informações de mais de um PDF, cada uma com sua fonte |
| 4 | Informação inexistente | "Qual o horário da biblioteca da FMU?" | "Não localizei essa informação nos documentos disponíveis." |
| 5 | Histórico da conversa | Após o teste 1: "E se eu ultrapassar esse limite?" | Entende que a pergunta se refere ao MEI |
| 6 | Novo documento | Copie um PDF para `Documents/` e clique em **Atualizar documentos** | O arquivo aparece na barra lateral e passa a ser consultado |
| 7 | Nova conversa | Clique em **Nova conversa** na barra lateral | O histórico é apagado |
| 8 | Responsividade | Reduza a janela ou abra no celular | Layout se adapta à tela |

Também é possível verificar mensagens de erro: sem `GROQ_API_KEY` no `.env`, o app exibe
um aviso e desabilita o campo de mensagem; com a pasta `Documents/` vazia, pede para
adicionar PDFs.

## Configurações opcionais

Além de `GROQ_API_KEY`, o `.env` aceita:

| Variável | Padrão | Descrição |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Modelo da Groq usado nas respostas |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Modelo de embeddings do Hugging Face (ao trocar, o índice é recriado) |
| `DOCS_DIR` | `Documents/` | Pasta com os PDFs |
| `CHROMA_DIR` | `chroma_db/` | Pasta do índice vetorial |

## Como funciona

1. Os PDFs de `Documents/` são lidos página por página.
2. O texto é dividido em trechos (*chunks*) de até 1000 caracteres, com sobreposição de 200.
   Cada trecho guarda o nome do arquivo e o número da página.
3. Cada trecho é convertido em um vetor (*embedding*) e salvo no ChromaDB.
4. Quando o usuário pergunta, os 8 trechos mais parecidos com a pergunta são buscados em
   todos os PDFs.
5. Os trechos, o histórico da conversa e a pergunta são enviados ao LLM da Groq, que
   responde usando somente esse conteúdo e citando as fontes.

## Solução de problemas

| Problema | Solução |
|---|---|
| "Chave da API Groq não configurada" | Verifique se o arquivo se chama `.env` (não `.env.txt`) e reinicie o app |
| "A chave da API Groq é inválida" | Gere uma nova chave em <https://console.groq.com/keys> |
| "Limite de uso da API Groq atingido" | Aguarde alguns instantes e tente novamente |
| PDF listado como "sem texto extraível" | O PDF é digitalizado como imagem; use uma versão com texto selecionável |
| Respostas desatualizadas após trocar PDFs | Clique em **Atualizar documentos** ou apague a pasta `chroma_db/` e reinicie |

## Limitações

- PDFs digitalizados (somente imagem) não são lidos, pois o projeto não usa OCR.
- Em documentos muito extensos, perguntas muito genéricas podem não recuperar o trecho
  ideal; perguntas mais específicas produzem melhores resultados.
- A indexação de PDFs grandes pode levar alguns minutos na primeira vez.
