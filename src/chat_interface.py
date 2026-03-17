import gradio as gr
from typing import List, Tuple, Dict
from .state_machine import StateMachine, UserIntent, IntentType
from .code_executor import CodeExecutor
from .model_loader import ModelLoader
import json
import os
from datetime import datetime


class ChatInterface:
    """Interfaz de chat estilo ChatGPT con máquina de estados"""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: str = None,
        history_dir: str = "./chat_history"
    ):
        self.state_machine = StateMachine()
        self.code_executor = CodeExecutor(timeout=5, max_output=10000)
        self.model_loader = ModelLoader(model_name, adapter_path)
        self.history_dir = history_dir
        self.current_chat_id = None
        self.chat_history = {}
        
        os.makedirs(history_dir, exist_ok=True)
        self._load_all_chats()
    
    def _load_all_chats(self):
        """Carga todos los chats guardados"""
        if os.path.exists(self.history_dir):
            for filename in os.listdir(self.history_dir):
                if filename.endswith('.json'):
                    chat_id = filename[:-5]
                    try:
                        with open(os.path.join(self.history_dir, filename), 'r', encoding='utf-8') as f:
                            self.chat_history[chat_id] = json.load(f)
                    except:
                        continue
    
    def _save_chat(self, chat_id: str, messages: List[Tuple[str, str]]):
        """Guarda un chat en disco"""
        chat_data = {
            'id': chat_id,
            'messages': messages,
            'created_at': datetime.now().isoformat(),
            'title': messages[0][0][:50] + '...' if messages else 'Nuevo Chat'
        }
        
        filepath = os.path.join(self.history_dir, f"{chat_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
    
    def _generate_chat_id(self) -> str:
        """Genera un ID único para el chat"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def create_new_chat(self) -> Tuple[List[Tuple[str, str]], str, str]:
        """Crea un nuevo chat"""
        self.current_chat_id = self._generate_chat_id()
        self.state_machine.conversation_history = []
        return [], self.current_chat_id, "Nuevo Chat"
    
    def load_chat(self, chat_id: str) -> Tuple[List[Tuple[str, str]], str, str]:
        """Carga un chat existente"""
        if chat_id and chat_id in self.chat_history:
            self.current_chat_id = chat_id
            chat_data = self.chat_history[chat_id]
            messages = chat_data.get('messages', [])
            title = chat_data.get('title', 'Chat')
            return messages, chat_id, title
        return [], chat_id, "Chat"
    
    def delete_chat(self, chat_id: str) -> Tuple[str, List[str]]:
        """Elimina un chat"""
        if chat_id and chat_id in self.chat_history:
            filepath = os.path.join(self.history_dir, f"{chat_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            del self.chat_history[chat_id]
            
            if self.current_chat_id == chat_id:
                self.current_chat_id = None
            
            chat_list = self._get_chat_list()
            return "Chat eliminado", chat_list
        return "", self._get_chat_list()
    
    def _get_chat_list(self) -> List[str]:
        """Obtiene la lista de chats guardados"""
        chats = []
        for chat_id, data in sorted(self.chat_history.items(), 
                                   key=lambda x: x[1].get('created_at', ''), 
                                   reverse=True):
            title = data.get('title', 'Sin título')[:40]
            chats.append(f"{chat_id}|{title}")
        return chats
    
    def process_message(
        self, 
        message: str, 
        history: List[Tuple[str, str]],
        chat_id: str
    ) -> Tuple[str, List[Tuple[str, str]], str, str]:
        """Procesa un mensaje del usuario"""
        if not message.strip():
            return "", history, chat_id, "Chat"
        
        if not chat_id:
            chat_id = self._generate_chat_id()
            self.current_chat_id = chat_id
        
        user_intent = self.state_machine.process_prompt(message)
        
        if user_intent.original_prompt.startswith("[BLOCKED]"):
            error_msg = "Lo siento, no puedo procesar esa solicitud por razones de seguridad."
            history.append((message, error_msg))
            self._save_chat(chat_id, history)
            return "", history, chat_id, "Chat"
        
        system_prompt = self.state_machine.get_system_prompt(user_intent)
        
        try:
            response = self.model_loader.generate(
                message,
                system_prompt,
                max_tokens=512
            )
        except Exception as e:
            response = "Lo siento, ocurrió un error al generar la respuesta. Por favor intentá de nuevo."
        
        code_blocks = self.code_executor.extract_code_blocks(
            response, 
            user_intent.language
        )
        
        execution_results = []
        if code_blocks and user_intent.language == "python":
            for code in code_blocks:
                if len(code) > 10:
                    result = self.code_executor.execute(code, user_intent.language)
                    execution_results.append(self.code_executor.format_result(result))
        
        if execution_results:
            response += "\n\n---\n\n" + "\n\n".join(execution_results)
        
        history.append((message, response))
        
        if len(history) == 1:
            title = message[:50] + '...' if len(message) > 50 else message
        else:
            title = "Chat"
        
        self._save_chat(chat_id, history)
        
        return "", history, chat_id, title
    
    def clear_chat(self) -> Tuple[List[Tuple[str, str]], str, str]:
        """Limpia el chat actual"""
        if self.current_chat_id:
            self._save_chat(self.current_chat_id, [])
        self.state_machine.conversation_history = []
        return [], self.current_chat_id, "Chat"
    
    def create_interface(self) -> gr.Blocks:
        """Crea la interfaz Gradio"""
        with gr.Blocks(
            title="Tutor de Algoritmia",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate"
            )
        ) as interface:
            
            gr.Markdown("""
            # 🎓 Tutor de Algoritmia
            
            Tu asistente personal para aprender algoritmos y estructuras de datos.
            """)
            
            with gr.Row():
                with gr.Column(scale=1, min_width=250):
                    gr.Markdown("### 💬 Historial")
                    
                    chat_list = gr.Dropdown(
                        label="Chats guardados",
                        choices=self._get_chat_list(),
                        value=None,
                        interactive=True
                    )
                    
                    with gr.Row():
                        new_chat_btn = gr.Button("➕ Nuevo", variant="secondary", size="sm")
                        delete_chat_btn = gr.Button("🗑️ Eliminar", variant="stop", size="sm")
                    
                    gr.Markdown("""
                    ---
                    ### 💡 Ejemplos
                    
                    - ¿Qué es Big O?
                    - Implementa una pila
                    - Explica BFS vs DFS
                    - Revisa mi código
                    """)
                    
                    status_box = gr.Textbox(
                        label="Estado",
                        value="",
                        interactive=False,
                        visible=False
                    )
                
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=550,
                        bubble_full_width=False,
                        show_copy_button=True,
                        avatar_images=(
                            None,
                            "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"
                        )
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Tu pregunta",
                            placeholder="Escribí tu pregunta sobre algoritmia...",
                            lines=2,
                            scale=4,
                            container=False
                        )
                        send_btn = gr.Button("Enviar", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Limpiar Chat")
                        export_btn = gr.Button("📤 Exportar")
                    
                    chat_id_box = gr.Textbox(visible=False)
                    title_box = gr.Textbox(label="Título", value="Chat", interactive=False)
            
            chat_list.change(
                fn=self.load_chat,
                inputs=[chat_list],
                outputs=[chatbot, chat_id_box, title_box]
            )
            
            new_chat_btn.click(
                fn=self.create_new_chat,
                outputs=[chatbot, chat_id_box, title_box]
            )
            
            delete_chat_btn.click(
                fn=self.delete_chat,
                inputs=[chat_id_box],
                outputs=[status_box, chat_list]
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot, chat_id_box, title_box]
            )
            
            export_btn.click(
                fn=self._export_chat,
                inputs=[chat_id_box, chatbot],
                outputs=[msg_input]
            )
            
            send_btn.click(
                fn=self.process_message,
                inputs=[msg_input, chatbot, chat_id_box],
                outputs=[msg_input, chatbot, chat_id_box, title_box]
            )
            
            msg_input.submit(
                fn=self.process_message,
                inputs=[msg_input, chatbot, chat_id_box],
                outputs=[msg_input, chatbot, chat_id_box, title_box]
            )
        
        return interface
    
    def _export_chat(self, chat_id: str, messages: List[Tuple[str, str]]) -> str:
        """Exporta el chat a texto plano"""
        if not messages:
            return "No hay mensajes para exportar"
        
        export_text = f"Chat de Algoritmia - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        export_text += "=" * 60 + "\n\n"
        
        for user_msg, assistant_msg in messages:
            export_text += f"Usuario: {user_msg}\n\n"
            export_text += f"Tutor: {assistant_msg}\n\n"
            export_text += "-" * 60 + "\n\n"
        
        export_path = os.path.join(self.history_dir, f"{chat_id}_export.txt")
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(export_text)
        
        return f"Chat exportado a: {export_path}"
    
    def launch(self, share: bool = False, server_port: int = 7860):
        """Lanza la interfaz"""
        self.model_loader.load()
        interface = self.create_interface()
        interface.launch(share=share, server_port=server_port)