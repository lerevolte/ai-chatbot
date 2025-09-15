from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain

from models.index import ChatMessage

FAISS_PATH = "/home/ai-chatbot/db_faiss_products_v1"

model = OllamaLLM(
    model="tinyllama",
    temperature=0.1,
    stop=["\nHuman:", "User:", "[INST]", "Вопрос:"]
)

embedding_function = OllamaEmbeddings(model="mxbai-embed-large")
db = FAISS.load_local(FAISS_PATH, embedding_function, allow_dangerous_deserialization=True)
chat_history = {}


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Вы — ИИ-ассистент для интернет-магазина. Ваша задача — отвечать на вопросы пользователя, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном ниже тексте. Не используйте никакие другие знания. Будьте кратки и точны."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """
Контекст из базы данных:
---
{context}
---
Основываясь строго на тексте выше, ответьте на следующий вопрос. Если в тексте нет ответа, скажите: "К сожалению, я не нашел точной информации по вашему вопросу."

Вопрос: {question}
"""
        ),
    ]
)

document_chain = create_stuff_documents_chain(llm=model, prompt=prompt_template)


def stream_rag_query(message: ChatMessage, session_id: str = ""):
    """
    Query a RAG system using streaming to provide faster perceived response times.
    """
    if session_id not in chat_history:
        chat_history[session_id] = []
    
    # ИЗМЕНЕНО: Возвращаемся к более простому и надежному ретриверу
    retriever = db.as_retriever(search_kwargs={'k': 4})
    
    context_docs = retriever.invoke(message.question)

    # ИЗМЕНЕНО: Добавлено логирование для отладки.
    # Теперь в консоли будет видно, какой контекст получает модель.
    print("\n--- CONTEXT DOCUMENTS RETRIEVED ---")
    if not context_docs:
        print("No documents found.")
    else:
        for i, doc in enumerate(context_docs):
            print(f"--- Document {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            # Печатаем первые 300 символов для краткости
            print(f"Content: {doc.page_content[:300]}...")
    print("-----------------------------------\n")

    response_stream = document_chain.stream({
        "context": context_docs,
        "question": message.question,
        "chat_history": chat_history[session_id]
    })

    full_response = ""
    for chunk in response_stream:
        yield chunk
        full_response += chunk

    chat_history[session_id].append(HumanMessage(content=message.question))
    chat_history[session_id].append(AIMessage(content=full_response))

