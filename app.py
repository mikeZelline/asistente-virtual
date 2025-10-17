import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env (si existe)
load_dotenv('config.env')

# Configura variables de entorno ANTES de los imports
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "mi-usuario-personalizado/0.0.1")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")



from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

app = Flask(__name__)

# Variables globales para el sistema RAG
vector_store = None
llm = None
graph = None
prompt_context = """Eres un asistente experto en contratación pública colombiana. 
Tu trabajo es responder preguntas basándote ÚNICAMENTE en la información que te proporciono.
Usa un tono amable y profesional, explicando en términos sencillos.
IMPORTANTE: Si el contexto contiene información relevante, DEBES responder con esa información.
No menciones que la información se está extrayendo de un documento pdf."""

# Estado (siguiendo patrón de LangChain)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Función para recuperar documentos relevantes
def retrieve(state: State):
    """Recupera los documentos más relevantes para la pregunta."""
    retrieved_docs_with_scores = vector_store.similarity_search_with_score(state["question"], k=3)
    
    # Debug para ver scores
    print(f"\n{'='*60}")
    print(f"[DEBUG] Pregunta: '{state['question']}'")
    print(f"[DEBUG] Scores obtenidos: {[round(score, 3) for _, score in retrieved_docs_with_scores]}")
    
    # Umbral más permisivo
    SIMILARITY_THRESHOLD = 1.5
    
    relevant_docs = []
    for doc, score in retrieved_docs_with_scores:
        print(f"[DEBUG] Evaluando doc con score {round(score, 3)} vs threshold {SIMILARITY_THRESHOLD}")
        if score < SIMILARITY_THRESHOLD:
            relevant_docs.append(doc)
    
    # Si no hay documentos relevantes pero hay resultados, tomar el mejor
    if not relevant_docs and retrieved_docs_with_scores:
        print(f"[DEBUG] Ningún documento bajo threshold, tomando el mejor score")
        best_doc, best_score = retrieved_docs_with_scores[0]
        # Solo si no es completamente irrelevante (score < 2.0)
        if best_score < 2.0:
            relevant_docs.append(best_doc)
            print(f"[DEBUG] Aceptando mejor documento con score {round(best_score, 3)}")
    
    print(f"[DEBUG] Documentos relevantes FINALES: {len(relevant_docs)}")
    print(f"{'='*60}\n")
    
    return {"context": relevant_docs}

# Función para generar respuesta
def generate(state: State):
    """Genera respuesta inteligente usando OpenAI basada en el contexto recuperado."""
    
    print(f"[DEBUG GENERATE] Cantidad de documentos en contexto: {len(state['context'])}")
    
    # Usar prompt_context cuando no hay documentos relevantes
    if not state["context"]:
        print(f"[DEBUG GENERATE] Sin documentos - enviando mensaje de rechazo")
        no_results_prompt = f"""Eres un asistente virtual especializado en contratación pública colombiana y el SECOP.

La pregunta del usuario es: "{state['question']}"

Esta pregunta NO está relacionada con tu especialización.

Responde de forma muy breve y amable que lamentablemente no puedes responder a esa pregunta específica, pero que estarás encantado de ayudar con cualquier tema relacionado con contratación pública colombiana o el SECOP.

IMPORTANTE: NO menciones documentos, archivos PDF, información proporcionada, o contextos. Simplemente di que ese tema no es tu especialidad pero que sí conoces sobre contratación pública."""
        
        try:
            response = llm.invoke(no_results_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            print(f"[DEBUG GENERATE] Respuesta de rechazo: {answer[:100]}...")
            return {"answer": answer.strip()}
        except Exception as e:
            print(f"[DEBUG GENERATE] Error en rechazo: {e}")
            return {"answer": "Lamentablemente no puedo responder esa pregunta, pero estaré encantado de ayudarte con temas relacionados con contratación pública colombiana o el SECOP."}
    
    # Si hay contexto relevante, generar respuesta basada en el documento
    print(f"[DEBUG GENERATE] CON documentos - generando respuesta basada en contexto")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    context_limit = 2000
    context = docs_content[:context_limit].strip()
    
    print(f"[DEBUG GENERATE] Longitud del contexto: {len(context)} caracteres")
    print(f"[DEBUG GENERATE] Primeros 100 chars del contexto: {context[:100]}...")
    
    prompt = f"""{prompt_context}

INFORMACIÓN RELEVANTE:
{context}

PREGUNTA: {state['question']}

Responde la pregunta de forma clara, precisa y profesional. No menciones que consultas documentos o información externa - simplemente responde como un experto que conoce el tema:"""
    
    print(f"[DEBUG GENERATE] Invocando OpenAI...")
    try:
        response = llm.invoke(prompt)
        print(f"[DEBUG GENERATE] Respuesta recibida de OpenAI")
        print(f"[DEBUG GENERATE] Tipo de respuesta: {type(response)}")
        
        answer = response.content if hasattr(response, 'content') else str(response)
        answer = answer.strip()
        
        print(f"[DEBUG GENERATE] Respuesta extraída: {answer[:150]}...")
        
        # Limitar longitud si es muy larga
        if len(answer) > 3000:
            answer = answer[:3000]
            if '.' in answer:
                answer = answer[:answer.rfind('.')+1]
        
        print(f"[DEBUG GENERATE] Respuesta final (longitud: {len(answer)})")
    
    except Exception as e:
        print(f"[DEBUG GENERATE] ERROR al invocar OpenAI: {e}")
        answer = f"Error al generar respuesta: {str(e)}"
    
    return {"answer": answer}

# Inicializar el sistema
def initialize_system():
    global vector_store, llm, graph
    
    print("Inicializando sistema...")
    
    # Cargar PDF
    loader = PyPDFLoader("fuente.pdf")
    docs = loader.load()
    
    # Dividir texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    
    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector store
    vector_store = FAISS.from_documents(all_splits, embedding_model)
    
    # Configurar OpenAI
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Modelo rápido y económico
        temperature=0.3,
    )
    
    # Compilar grafo
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    print(f"Sistema inicializado: {len(docs)} páginas, {len(all_splits)} fragmentos.")

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para procesar preguntas
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Pregunta vacía'}), 400
        
        # Invocar el grafo
        initial_state = State(question=question, context=[], answer="")
        final_state = graph.invoke(initial_state)
        
        return jsonify({
            'answer': final_state['answer']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_system()
    
    # Obtener configuración desde variables de entorno
    debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
    port = int(os.getenv("PORT", "5000"))
    
    print("\n" + "="*60)
    print(f"Servidor web iniciado en http://localhost:{port}")
    print(f"Modo Debug: {debug_mode}")
    print("="*60 + "\n")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

