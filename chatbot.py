import json
import numpy as np
import re
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import os
import warnings
from functools import lru_cache

# Configurar el paralelismo del tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Filtrar advertencias específicas
warnings.filterwarnings("ignore", category=FutureWarning)

# Configurar logging
logging.basicConfig(
    filename=f'chatbot_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ChatBot:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.app = Flask(__name__)
        self.model_name = model_name
        self.model = None
        self.faqs = []
        self.faq_questions = []
        self.faq_embeddings = None
        self.unrecognized_questions = []
        self.similarity_threshold = 0.6  # Ajustado para mejorar la respuesta
        self.cache_size = 1000  # Tamaño del caché
        self.setup_routes()
        self.initialize_model()
        self.load_faqs()

    def initialize_model(self):
        """Inicializa el modelo de manera segura con configuración optimizada"""
        try:
            # Configurar CUDA si está disponible
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = SentenceTransformer(self.model_name)
            self.model.to(device)
            logging.info(f"Modelo {self.model_name} cargado exitosamente en {device}")
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {e}")
            raise

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> torch.Tensor:
        """Cachea los embeddings para mejorar el rendimiento"""
        return self.model.encode(text, convert_to_tensor=True)

    def load_faqs(self, file_path: str = 'faq.json'):
        """Carga y preprocesa las FAQs con manejo de memoria optimizado"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.faqs = json.load(file)
            
            # Procesar preguntas por lotes para mejor rendimiento
            batch_size = 32
            self.faq_questions = [self.preprocess_text(faq['question']) for faq in self.faqs]
            
            # Generar embeddings por lotes
            embeddings_list = []
            for i in range(0, len(self.faq_questions), batch_size):
                batch = self.faq_questions[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
                embeddings_list.append(batch_embeddings)
            
            self.faq_embeddings = torch.cat(embeddings_list, dim=0)
            logging.info(f"FAQs cargadas exitosamente: {len(self.faqs)} preguntas")
        except Exception as e:
            logging.error(f"Error al cargar FAQs: {e}")
            self.faqs = []
            self.faq_questions = []
            self.faq_embeddings = None

    def preprocess_text(self, text: str) -> str:
        """Preprocesa el texto de manera eficiente"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\u00C0-\u00FF?!.,]', '', text)  # Permitir signos de puntuación 
        return ' '.join(text.split())

    @lru_cache(maxsize=1000)
    def find_best_answer(self, question: str) -> Tuple[Optional[str], float]:
        """Encuentra la mejor respuesta con caché"""
        try:
            processed_question = self.preprocess_text(question)
            question_embedding = self.get_embedding(processed_question)
            
            # Calcular similitudes 
            with torch.no_grad():  # Desactivar gradientes para mejor rendimiento
                similarities = util.pytorch_cos_sim(question_embedding, self.faq_embeddings)
            
            best_match_idx = similarities.argmax()
            confidence_score = similarities[0][best_match_idx].item()

            if confidence_score > self.similarity_threshold:
                answer = self.faqs[best_match_idx]['answer']
                logging.info(f"Pregunta: {question} | Score: {confidence_score:.2f}")
                return answer, confidence_score
            else:
                self.log_unrecognized_question(question, confidence_score)
                return None, confidence_score
        except Exception as e:
            logging.error(f"Error al procesar pregunta: {question} | Error: {e}")
            return None, 0.0

    def log_unrecognized_question(self, question: str, score: float):
        """Registra preguntas no reconocidas de manera segura"""
        try:
            self.unrecognized_questions.append({
                'question': question,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })
            logging.warning(f"Pregunta no reconocida: {question} | Score: {score:.2f}")
        except Exception as e:
            logging.error(f"Error al registrar pregunta no reconocida: {e}")

    def setup_routes(self):
        """Configura las rutas de Flask con manejo de errores mejorado"""
        @self.app.route('/')
        def home():
            return render_template('index.html')

        @self.app.route('/ask', methods=['GET'])
        def ask():
            try:
                question = request.args.get('question', '').strip()
                if not question:
                    return jsonify({
                        'answer': 'Please ask a question.',
                        'confidence': 0.0,
                        'status': 'error'
                    })

                answer, confidence = self.find_best_answer(question)
                
                response = {
                    'answer': answer if answer else 'Sorry, I dont have an answer to that question..',
                    'confidence': round(confidence, 2),
                    'status': 'success' if answer else 'no_match'
                }

                # Mejorar la lógica de sugerencias
                if confidence <= self.similarity_threshold:
                    similar_questions = self.get_similar_questions(question)
                    if similar_questions:
                        response['suggestions'] = similar_questions

                return jsonify(response)
            except Exception as e:
                logging.error(f"Error en la ruta /ask: {e}", exc_info=True)
                return jsonify({
                    'answer': 'Sorry, an error occurred while processing your question.',
                    'confidence': 0.0,
                    'status': 'error'
                })

    def get_similar_questions(self, question: str) -> List[str]:
        """Obtiene preguntas similares a la pregunta dada"""
        try:
            processed_question = self.preprocess_text(question)
            question_embedding = self.get_embedding(processed_question)
            with torch.no_grad():
                similarities = util.pytorch_cos_sim(question_embedding, self.faq_embeddings)
            
            similar_indices = torch.topk(similarities, k=5).indices[0].tolist()
            similar_questions = [self.faq_questions[idx] for idx in similar_indices if idx != similarities.argmax()]
            return similar_questions
        except Exception as e:
            logging.error(f"Error al obtener preguntas similares: {e}")
            return []

    def run(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = False):
        """Inicia el servidor Flask con configuración optimizada"""
        self.app.run(
            host=host, 
            port=port, 
            debug=debug,
            threaded=True,  # Habilitar multi-threading
            use_reloader=False if debug else None  # Evitar problemas con el reloader en modo debug
        )

if __name__ == '__main__':
    chatbot = ChatBot()
    chatbot.run(debug=True, port=5001)
