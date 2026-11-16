"""
Assistente Virtual do Centro Universitário FMU (interface Streamlit).

Para executar:  python -m streamlit run app.py
Para parar:     Ctrl + C no terminal

A lógica de documentos/RAG fica em rag.py; este arquivo cuida apenas da
interface, do histórico da conversa e do tratamento de erros para o usuário.
"""

import html
import os
import threading
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 1. CONFIGURAÇÃO ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Carrega o .env da pasta do projeto (independe do diretório de execução).
# Deve acontecer antes de importar rag.py, que lê variáveis opcionais.
load_dotenv(BASE_DIR / ".env")

import rag  # noqa: E402

st.set_page_config(
    page_title="Assistente Virtual | Centro Universitário FMU",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="auto",
)

AVATARS = {"user": ":material/person:", "assistant": ":material/school:"}


# 2. COMPONENTES VISUAIS -----------------------------------------------------

def load_css() -> None:
    css_path = BASE_DIR / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <header class="fmu-header" role="banner">
          <div class="fmu-logo" aria-hidden="true">FMU</div>
          <div class="fmu-header-text">
            <p class="fmu-eyebrow">Assistente Virtual</p>
            <h1>Centro Universitário FMU</h1>
            <p class="fmu-subtitle">Tire suas dúvidas com base nos documentos oficiais disponíveis.</p>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_welcome(total_docs: int) -> None:
    plural = "documento disponível" if total_docs == 1 else "documentos disponíveis"
    st.markdown(
        f"""
        <section class="fmu-welcome" aria-label="Boas-vindas">
          <h2>Olá! Como posso ajudar?</h2>
          <p>Sou o assistente virtual do <strong>Centro Universitário FMU</strong>.
          Respondo às suas perguntas consultando <strong>{total_docs} {plural}</strong>.</p>
          <ul>
            <li>As respostas se baseiam <strong>somente</strong> no conteúdo dos documentos.</li>
            <li>Sempre que possível, indico o <strong>arquivo e a página</strong> de origem.</li>
            <li>Se a informação não estiver nos documentos, eu aviso.</li>
          </ul>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_notice(title: str, body: str) -> None:
    st.markdown(
        f'<div class="fmu-notice" role="alert"><strong>{title}</strong><br>{body}</div>',
        unsafe_allow_html=True,
    )


def typing_indicator(text: str) -> str:
    return (
        '<div class="fmu-typing" role="status" aria-live="polite">'
        '<span class="fmu-dots" aria-hidden="true"><span></span><span></span><span></span></span>'
        f"{html.escape(text)}</div>"
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    chips = []
    for src in sources:
        pages = ", ".join(str(p) for p in src["pages"])
        label = f"{src['file']} · p. {pages}"
        chips.append(f'<span class="fmu-source-chip" title="{html.escape(label)}">'
                     f"📄 {html.escape(label)}</span>")
    st.markdown(
        '<div class="fmu-sources"><span class="fmu-sources-label">Trechos consultados:</span>'
        + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def render_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    with st.chat_message(role, avatar=AVATARS[role]):
        # Marcador invisível usado pelo CSS para estilizar o balão de cada autor.
        st.markdown(f'<span class="msg-marker msg-{role}"></span>', unsafe_allow_html=True)
        st.markdown(content)
        render_sources(sources or [])


# 3. RECURSOS EM CACHE (criados uma única vez por processo) -------------------

@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def get_vectorstore():
    return rag.open_vectorstore(rag.create_embeddings())


@st.cache_resource(show_spinner=False)
def get_llm():
    return rag.create_llm()


@st.cache_resource(show_spinner=False)
def get_index_state() -> dict:
    """Estado compartilhado entre sessões: evita sincronizações simultâneas."""
    return {"signature": None, "report": None, "lock": threading.Lock()}


def ensure_index(force: bool = False) -> rag.IndexReport:
    """Sincroniza o índice somente quando a pasta Documents muda.

    A assinatura (nome/tamanho/data) é barata de calcular; os embeddings só
    são gerados para PDFs novos ou alterados, e ficam salvos em disco.
    """
    store = get_vectorstore()
    state = get_index_state()
    signature = rag.folder_signature()

    with state["lock"]:
        if force or state["signature"] != signature or state["report"] is None:
            with st.status("Preparando os documentos...", expanded=False) as status:
                bar = st.progress(0.0)

                def progress(message: str, fraction: float) -> None:
                    status.update(label=message)
                    bar.progress(min(max(fraction, 0.0), 1.0))

                report = rag.sync_index(store, progress=progress)
                status.update(label="Documentos prontos", state="complete")
            state["signature"] = signature
            state["report"] = report
        return state["report"]


def friendly_llm_error(exc: Exception) -> str:
    """Traduz erros da API da Groq em mensagens compreensíveis."""
    try:
        import groq
    except ImportError:  # pragma: no cover
        groq = None
    if groq is not None:
        if isinstance(exc, groq.AuthenticationError):
            return "A chave da API Groq é inválida. Verifique a variável GROQ_API_KEY no arquivo .env."
        if isinstance(exc, groq.RateLimitError):
            return "Limite de uso da API Groq atingido. Aguarde alguns instantes e tente novamente."
        if isinstance(exc, groq.APIConnectionError):
            return "Não foi possível conectar à API Groq. Verifique sua conexão com a internet."
        if isinstance(exc, groq.APIStatusError):
            return f"A API Groq retornou um erro ({exc.status_code}). Tente novamente em instantes."
    return f"Ocorreu um erro ao gerar a resposta: {exc}"


# 4. INTERFACE -----------------------------------------------------------------

load_css()
render_header()

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []


def clear_chat() -> None:
    st.session_state.mensagens = []


# 4.1 Verificações iniciais (pasta de documentos e chave da API)
pdf_files = rag.list_pdfs()
has_api_key = bool(os.getenv("GROQ_API_KEY", "").strip())
report: rag.IndexReport | None = None
can_chat = True

if not rag.DOCS_DIR.is_dir() or not pdf_files:
    can_chat = False
    render_notice(
        "Nenhum documento encontrado.",
        f"Adicione arquivos PDF à pasta <code>{html.escape(rag.DOCS_DIR.name)}</code> "
        "do projeto e recarregue a página.",
    )
else:
    try:
        report = ensure_index()
    except Exception as exc:  # noqa: BLE001
        can_chat = False
        st.error(f"Erro ao preparar o índice de documentos: {exc}")

    if report is not None and not report.indexed_files:
        can_chat = False
        render_notice(
            "Não foi possível extrair texto dos PDFs.",
            "Verifique se os arquivos não estão corrompidos ou digitalizados apenas como imagem.",
        )

if not has_api_key:
    can_chat = False
    render_notice(
        "Chave da API Groq não configurada.",
        "Defina <code>GROQ_API_KEY</code> no arquivo <code>.env</code> do projeto "
        "e reinicie o aplicativo.",
    )

# 4.2 Barra lateral: documentos indexados e ações
with st.sidebar:
    st.markdown("### 📚 Documentos")
    if report and report.indexed_files:
        items = "".join(
            f'<div class="fmu-doc-item"><span class="fmu-doc-name">{html.escape(name)}</span>'
            f'<span class="fmu-doc-count">{count} trechos</span></div>'
            for name, count in sorted(report.indexed_files.items(), key=lambda i: i[0].lower())
        )
        st.markdown(items, unsafe_allow_html=True)
        st.caption(f"{len(report.indexed_files)} PDF(s) · {report.total_chunks} trechos indexados")
    else:
        st.caption("Nenhum documento indexado.")

    if report:
        for name in report.empty:
            st.warning(f"Sem texto extraível: {name}", icon="⚠️")
        for name, error in report.errors.items():
            st.error(f"Falha ao ler {name}: {error}", icon="🚫")

    st.divider()
    st.button("🗨️ Nova conversa", on_click=clear_chat, use_container_width=True)
    if st.button("🔄 Atualizar documentos", use_container_width=True,
                 help="Verifica a pasta Documents e indexa PDFs novos ou alterados."):
        if pdf_files:
            ensure_index(force=True)
        st.rerun()

# 4.3 Boas-vindas e histórico
welcome_slot = st.empty()
if not st.session_state.mensagens and report and report.indexed_files:
    with welcome_slot.container():
        render_welcome(len(report.indexed_files))

for msg in st.session_state.mensagens:
    render_message(msg["role"], msg["content"], msg.get("sources"))


# 5. PERGUNTA E RESPOSTA -------------------------------------------------------

def answer_question(question: str) -> None:
    # Histórico anterior à pergunta atual (usado para dar contexto ao LLM).
    history = list(st.session_state.mensagens)

    render_message("user", question)
    st.session_state.mensagens.append({"role": "user", "content": question})

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown('<span class="msg-marker msg-assistant"></span>', unsafe_allow_html=True)
        loader = st.empty()
        loader.markdown(typing_indicator("Consultando os documentos..."), unsafe_allow_html=True)

        llm = get_llm()

        # 5.1 Recuperação dos trechos relevantes
        try:
            search_query = rag.condense_question(llm, history, question)
            documents = rag.create_retriever(get_vectorstore()).invoke(search_query)
        except Exception as exc:  # noqa: BLE001
            loader.empty()
            st.error(f"Erro ao buscar nos documentos: {exc}")
            return

        # 5.2 Geração da resposta (streaming) com base nos trechos
        loader.markdown(typing_indicator("Elaborando a resposta..."), unsafe_allow_html=True)
        messages = rag.build_messages(history, question, documents)

        def stream():
            first = True
            for token in rag.stream_answer(llm, messages):
                if first:
                    loader.empty()  # remove o indicador ao chegar o primeiro trecho
                    first = False
                yield token

        try:
            answer = st.write_stream(stream())
        except Exception as exc:  # noqa: BLE001
            loader.empty()
            st.error(friendly_llm_error(exc))
            return
        loader.empty()

        answer = answer if isinstance(answer, str) else "".join(map(str, answer))
        if not answer.strip():
            answer = rag.NOT_FOUND_MESSAGE
            st.markdown(answer)

        # Só mostra os trechos consultados quando a resposta foi encontrada.
        sources = [] if rag.NOT_FOUND_MESSAGE in answer else rag.unique_sources(documents)
        render_sources(sources)

    st.session_state.mensagens.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


placeholder = "Digite sua pergunta..." if can_chat else "Assistente indisponível no momento"
if pergunta := st.chat_input(placeholder, disabled=not can_chat):
    pergunta = pergunta.strip()
    if pergunta:
        welcome_slot.empty()
        answer_question(pergunta)
