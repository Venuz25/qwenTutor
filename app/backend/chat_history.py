import json
import os
import sqlite3
from datetime import datetime
from app.config import DB_PATH

class ChatHistory:
    """Gestiona historial de chats con persistencia SQLite."""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or str(DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats (id)
            )
        ''')
        conn.commit()
        conn.close()
    
    def create_chat(self, title="Nuevo Chat"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO chats (title) VALUES (?)', (title,))
        chat_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return chat_id
    
    def add_message(self, chat_id, role, content):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)', (chat_id, role, content))
        cursor.execute('UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?', (chat_id,))
        conn.commit()
        conn.close()
    
    def get_chat_history(self, chat_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC', (chat_id,))
        messages = cursor.fetchall()
        conn.close()
        return [{'role': r, 'content': c} for r, c in messages]
    
    def get_all_chats(self, limit=50):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC LIMIT ?', (limit,))
        chats = cursor.fetchall()
        conn.close()
        return [{'id': cid, 'title': t, 'created_at': ca, 'updated_at': ua} for cid, t, ca, ua in chats]
    
    def delete_chat(self, chat_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
        cursor.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
        conn.commit()
        conn.close()
    
    def update_chat_title(self, chat_id, title):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?', (title, chat_id))
        conn.commit()
        conn.close()