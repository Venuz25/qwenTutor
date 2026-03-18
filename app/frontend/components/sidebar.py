import streamlit as st

def render_sidebar(history, current_chat, messages, on_new_chat, on_select_chat, on_delete_chat, on_open_compiler):
    """Renderiza la barra lateral con historial de chats."""
    
    with st.sidebar:
        st.title("🎓 Tutor de Algoritmia")
        st.subheader("📚 Historial")
        
        if st.button("➕ Nuevo Chat", use_container_width=True):
            on_new_chat()
        
        chats = history.get_all_chats(limit=20)
        for chat in chats:
            title = chat['title'] if chat['title'] else f"Chat #{chat['id']}"
            if st.button(f"💬 {title[:25]}...", key=f"chat_{chat['id']}"):
                on_select_chat(chat['id'])
        
        if current_chat and st.button("🗑️ Eliminar", use_container_width=True):
            on_delete_chat(current_chat)
        
        st.divider()
        st.subheader("ℹ️ Info")
        st.markdown("- **Modelo**: Qwen2.5-1.5B\n- **Fine-tuning**: QLoRA\n- **Dataset**: ~1000 ejemplos\n- **VRAM**: ~3.5 GB")
        
        st.divider()
        if st.button("⚙️ Compilador", use_container_width=True, type="primary"):
            on_open_compiler()