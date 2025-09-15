import hashlib
import os
import shutil
import time
from typing import List
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


# --- КОНФИГУРАЦИЯ ---
# Путь к директории с текстовыми файлами товаров
DATA_PATH = "./docs"
# Путь для сохранения векторной базы данных FAISS
FAISS_PATH = "./db_faiss_products_v1" 
# Название модели для эмбеддингов, которую вы используете в Ollama
# СОВЕТ: Для ускорения можно попробовать более легкую модель, например "nomic-embed-text"
EMBEDDING_MODEL = "mxbai-embed-large" 
# --- КОНЕЦ КОНФИГУРАЦИИ ---


def walk_through_files(path, file_extension='.txt'):
    """Проходит по всем файлам в указанной директории."""
    for (dir_path, dir_names, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)


def load_and_deduplicate_documents() -> List[Document]:
    """
    Загружает все документы из DATA_PATH и удаляет дубликаты по содержимому.
    """
    documents = []
    unique_hashes = set()

    print(f"Загрузка документов из '{DATA_PATH}'...")
    file_paths = list(walk_through_files(DATA_PATH))
    
    for f_name in tqdm(file_paths, desc="Чтение файлов"):
        try:
            document_loader = TextLoader(f_name, encoding="utf-8")
            doc = document_loader.load()[0] # TextLoader возвращает список из одного документа
            
            # Проверка на дубликаты по хэшу содержимого
            content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
            if content_hash not in unique_hashes:
                documents.append(doc)
                unique_hashes.add(content_hash)
        except Exception as e:
            print(f"Не удалось прочитать файл {f_name}: {e}")
            
    print(f"Загружено {len(documents)} уникальных документов.")
    return documents


def save_to_faiss(documents: List[Document]):
    """
    Создает и сохраняет векторную базу данных FAISS из документов.
    Обрабатывает документы пакетами, чтобы не перегружать сервис эмбеддингов.
    """
    if not documents:
        print("Нет документов для сохранения в базу данных.")
        return

    # Удаляем старую базу данных, если она существует
    if os.path.exists(FAISS_PATH):
        print(f"Удаление старой базы данных: {FAISS_PATH}")
        shutil.rmtree(FAISS_PATH)

    print(f"Инициализация модели эмбеддингов: {EMBEDDING_MODEL}")
    # ВАЖНО: Убедитесь, что Ollama настроена на использование GPU, если он у вас есть.
    # Это самый важный фактор ускорения.
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # ИЗМЕНЕНО: Значительно увеличен размер пакета.
    # Если у вас мощная видеокарта с большим объемом VRAM (>8GB), можно поставить 128 или 256.
    # Если вы используете CPU, большое значение также может ускорить процесс за счет меньшего количества вызовов.
    batch_size = 64
    db = None

    print("Создание векторов и сохранение в FAISS...")
    # Итерация по документам с использованием tqdm для отображения прогресс-бара
    for i in tqdm(range(0, len(documents), batch_size), desc="Обработка пакетов"):
        batch = documents[i:i+batch_size]
        if not batch:
            continue
        
        try:
            if db is None:
                # Создаем базу данных с первым пакетом
                db = FAISS.from_documents(batch, embedding_function)
            else:
                # Добавляем последующие пакеты в существующую базу
                db.add_documents(batch)
            
            # ИЗМЕНЕНО: Удалена искусственная задержка time.sleep(0.5).
            # Она не нужна при работе с локальным сервером Ollama и только замедляла процесс.
        except Exception as e:
            print(f"\nОшибка при обработке пакета {i//batch_size + 1}: {e}")
            print("Продолжение работы со следующего пакета...")
            continue
            
    if db:
        db.save_local(FAISS_PATH)
        print(f"\nБаза данных успешно сохранена. Обработано {len(documents)} документов.")
        print(f"Путь к базе: {FAISS_PATH}")
    else:
        print("\nНе удалось создать базу данных, так как не было обработано ни одного документа.")


def main():
    """
    Основная функция для генерации векторной базы данных.
    """
    # 1. Загружаем документы
    documents = load_and_deduplicate_documents()
    
    # 2. Сохраняем их в векторную базу
    save_to_faiss(documents)


if __name__ == "__main__":
    main()

