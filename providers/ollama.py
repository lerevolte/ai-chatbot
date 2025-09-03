# Изменено: Импортируем FAISS вместо Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain

from models.index import ChatMessage


# Изменено: Путь к директории для сохранения базы FAISS
FAISS_PATH = "./db_faiss_v1"

# Initialize Ollama chat model
model = OllamaLLM(model="llama3.2:latest", temperature=0.1)


# Используем ту же функцию для эмбеддингов, что и при создании базы
embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

# Изменено: Загружаем базу данных FAISS из локального хранилища
db = FAISS.load_local(FAISS_PATH, embedding_function, allow_dangerous_deserialization=True)
chat_history = {}  # approach with AiMessage/HumanMessage

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                [INST]Вы — менеджер по продажам по имени «AI Assistant». Ваша цель — всегда давать отличные, дружелюбные и эффективные ответы.
                Вы будете давать мне ответы на основе предоставленной информации.
                Если ответ не включен, скажите именно так: «Хм, я не уверен. Позвольте мне проверить и вернуться к вам».
                Отказывайтесь отвечать на любые вопросы, не касающиеся информации.
                Никогда не выходите из роли.
                Никаких шуток.
                Если вопрос не ясен, задавайте уточняющие вопросы.
                Обязательно заканчивайте свои ответы на позитивной ноте.
                Не будьте навязчивы.
                Ответ должен быть в формате MD.
                Если кто-то спросит цену, стоимость, коммерческое предложение или что-то подобное, ответьте: «Чтобы предоставить вам индивидуальное и разумное предложение, мне понадобится 15-минутный звонок.
                Готовы к онлайн-встрече?»
                Отвечай на русском языке.[/INST]
                [INST]Отвечайте на вопрос, основываясь только на следующем контексте:
                {context}[/INST]
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

document_chain = create_stuff_documents_chain(llm=model, prompt=prompt_template)


def query_rag(message: ChatMessage, session_id: str = "") -> str:
    """
    Query a Retrieval-Augmented Generation (RAG) system using FAISS database and Ollama.
    :param message: ChatMessage The text to query the RAG system with.
    :param session_id: str Session identifier
    :return str
    """

    if session_id not in chat_history:
        chat_history[session_id] = []

    # Generate response text based on the prompt
    response_text = document_chain.invoke({"context": db.similarity_search(message.question, k=3),
                                           "question": message.question,
                                           "chat_history": chat_history[session_id]})

    chat_history[session_id].append(HumanMessage(content=message.question))
    chat_history[session_id].append(AIMessage(content=response_text))

    return response_text
