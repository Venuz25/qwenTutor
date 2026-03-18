import streamlit as st

def render_compiler_modal(compiler, on_close):
    """Renderiza el modal del compilador."""
    
    st.markdown("---")
    st.subheader("🔧 Compilador en Línea")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        code = st.text_area("Código", value=compiler.get_language_template("python"), height=300, key="comp_code")
    
    with col2:
        language = st.selectbox("Lenguaje", compiler.get_supported_languages(), index=0, key="comp_lang")
        stdin = st.text_area("Entrada (stdin)", height=100, key="comp_stdin")
        
        if st.button("▶️ Ejecutar", type="primary", use_container_width=True):
            with st.spinner("Compilando..."):
                result = compiler.compile_code(
                    st.session_state.get('comp_code', ''),
                    st.session_state.get('comp_lang', 'python'),
                    st.session_state.get('comp_stdin', '')
                )
                st.session_state.comp_result = result
    
    if hasattr(st.session_state, 'comp_result'):
        result = st.session_state.comp_result
        status = "✅ Exitoso" if result['success'] else "❌ Error"
        st.markdown(f"**Estado:** {status}")
        if result['execution_time'] > 0:
            st.markdown(f"**Tiempo:** {result['execution_time']:.2f}s")
        if result['output']:
            st.markdown("### 📤 Salida")
            st.code(result['output'], language=st.session_state.get('comp_lang', 'python'))
        if result['error']:
            st.markdown("### ❌ Error")
            st.error(result['error'])
    
    if st.button("✕ Cerrar"):
        on_close()