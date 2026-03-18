import sys
from pathlib import Path

# Agregar raíz del proyecto al path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Agregar modules/ al path para imports directos
modules_path = project_root / "modules"
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))

# ================= IMPORTS DIRECTOS (sin 'app.') =================
import streamlit as st

# Importar directamente desde modules/
from model_loader import ModelLoader
from prompt_filter import PromptFilter
from code_compiler import CodeCompiler
from chat_history import ChatHistory

# Config desde raíz
from config import STREAMLIT_PORT, STREAMLIT_ADDRESS, STREAMLIT_THEME, ADAPTER_PATH

# ================= CONFIGURACIÓN DE PÁGINA =================
st.set_page_config(
    page_title="Tutor de Algoritmia",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS PERSONALIZADO =================
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1e2327; border-right: 1px solid #333; }
    .main { background-color: #0e1117; }
    .stChatMessage { background-color: #262730; border-radius: 10px; padding: 10px; margin: 5px 0; }
    .stButton > button { background-color: #0066cc; color: white; border-radius: 5px; border: none; padding: 8px 16px; }
    .stButton > button:hover { background-color: #0052a3; }
    .stTextInput > div > div > input { background-color: #262730; color: white; border: 1px solid #444; }
    h1, h2, h3 { color: #ffffff !important; }
    pre { background-color: #1e1e1e !important; border-radius: 5px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ================= INICIALIZAR ESTADO =================
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.prompt_filter = PromptFilter()
    st.session_state.compiler = CodeCompiler()
    st.session_state.history = ChatHistory()
    st.session_state.current_chat = None
    st.session_state.messages = []
    st.session_state.show_compiler = False

# ================= CARGAR MODELO (cache) =================
@st.cache_resource
def load_model_cached(adapter_path):
    loader = ModelLoader(adapter_path=adapter_path)
    loader.load_model()
    return loader

if not st.session_state.model_loaded:
    with st.spinner("🤖 Cargando modelo... (2-5 minutos)"):
        try:
            st.session_state.model = load_model_cached(str(ADAPTER_PATH))
            st.session_state.model_loaded = True
            st.success("✅ Modelo cargado")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🎓 Tutor de Algoritmia")
    st.subheader("📚 Historial")
    
    if st.button("➕ Nuevo Chat", use_container_width=True):
        st.session_state.current_chat = st.session_state.history.create_chat()
        st.session_state.messages = []
        st.rerun()
    
    chats = st.session_state.history.get_all_chats()
    for chat in chats:
        title = chat['title'] if chat['title'] else f"Chat #{chat['id']}"
        if st.button(f"💬 {title[:25]}...", key=f"chat_{chat['id']}"):
            st.session_state.current_chat = chat['id']
            st.session_state.messages = st.session_state.history.get_chat_history(chat['id'])
            st.rerun()
    
    if st.session_state.current_chat and st.button("🗑️ Eliminar", use_container_width=True):
        st.session_state.history.delete_chat(st.session_state.current_chat)
        st.session_state.current_chat = None
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.subheader("ℹ️ Info")
    st.markdown("- **Modelo**: Qwen2.5-1.5B\n- **Fine-tuning**: QLoRA\n- **Dataset**: ~1000 ejemplos\n- **VRAM**: ~3.5 GB")
    
    st.divider()
    if st.button("⚙️ Compilador", use_container_width=True, type="primary"):
        st.session_state.show_compiler = True

# ================= ÁREA PRINCIPAL =================
st.title("💬 Chat con el Tutor")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Preguntá sobre algoritmia, estructuras de datos, o pedí ayuda con código..."):
    if st.session_state.current_chat is None:
        st.session_state.current_chat = st.session_state.history.create_chat(
            title=prompt[:50] + "..." if len(prompt) > 50 else prompt
        )
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.history.add_message(st.session_state.current_chat, "user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    processed_prompt, metadata = st.session_state.prompt_filter.process_prompt(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Pensando..."):
            try:
                system_message = {"role": "system", "content": "Eres un tutor experto en Algoritmia, Estructuras de Datos y Programación Competitiva. No des respuestas directas. Guiá con preguntas socráticas, pistas y explicaciones paso a paso."}
                messages_for_model = [system_message] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                response = st.session_state.model.generate_response(messages_for_model)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.history.add_message(st.session_state.current_chat, "assistant", response)
            except Exception as e:
                st.error(f"Error: {e}")

# ================= MODAL COMPILADOR =================
if st.session_state.get('show_compiler', False):
    st.markdown("---")
    st.subheader("🔧 Compilador en Línea")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        code = st.text_area("Código", value=st.session_state.compiler.get_language_template("python"), height=300, key="compiler_code")
    
    with col2:
        language = st.selectbox("Lenguaje", st.session_state.compiler.get_supported_languages(), index=0, key="compiler_lang")
        stdin = st.text_area("Entrada (stdin)", height=100, key="compiler_stdin")
        
        if st.button("▶️ Ejecutar", type="primary", use_container_width=True):
            with st.spinner("Compilando..."):
                result = st.session_state.compiler.compile_code(
                    st.session_state.compiler_code,
                    st.session_state.compiler_lang,
                    st.session_state.compiler_stdin
                )
                st.session_state.compiler_result = result
    
    if hasattr(st.session_state, 'compiler_result'):
        result = st.session_state.compiler_result
        status = "✅ Exitoso" if result['success'] else "❌ Error"
        st.markdown(f"**Estado:** {status}")
        if result['execution_time'] > 0:
            st.markdown(f"**Tiempo:** {result['execution_time']:.2f}s")
        if result['output']:
            st.markdown("### 📤 Salida")
            st.code(result['output'], language=st.session_state.compiler_lang)
        if result['error']:
            st.markdown("### ❌ Error")
            st.error(result['error'])
    
    if st.button("✕ Cerrar Compilador"):
        st.session_state.show_compiler = False
        st.rerun()

# ================= FOOTER =================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 0.9em;'>🎓 Tutor de Algoritmia • Qwen2.5-1.5B fine-tuned con QLoRA</div>", unsafe_allow_html=True)