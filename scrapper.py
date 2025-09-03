from langchain_community.document_transformers import (
    Html2TextTransformer,
    BeautifulSoupTransformer,
)
from langchain_core.documents import Document
import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import Union
from tqdm.asyncio import tqdm_asyncio
import random


FILE_TO_PARSE = "data/links.txt"
DIR_TO_STORE = "docs"


def getLinks2Parse() -> list:
    try:
        with open(FILE_TO_PARSE, "r") as f:
            return [link.strip() for link in f.readlines()]
    except:
        return []

async def _fetch_and_create_doc(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore, retries=4, backoff_factor=2.0) -> Union[Document, None]:
    """
    Асинхронно загружает одну страницу с несколькими попытками и экспоненциальной задержкой.
    """
    # Ждем, пока семафор не освободится, чтобы ограничить количество одновременных запросов
    async with semaphore:
        for i in range(retries):
            try:
                # Добавляем небольшую случайную задержку перед каждым запросом, чтобы имитировать поведение человека
                await asyncio.sleep(random.uniform(1.0, 3.0))
                async with session.get(url, ssl=False, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        return Document(page_content=content, metadata={"source": url})
                    # Обрабатываем ошибку 503 (Service Unavailable)
                    elif response.status == 503:
                        delay = backoff_factor * (2 ** i)
                        print(f"Ошибка 503 для {url}. Повторная попытка через {delay:.1f} сек. (Попытка {i + 1}/{retries})")
                        await asyncio.sleep(delay)
                    else:
                        print(f"Ошибка при загрузке {url}: статус {response.status}")
                        return None  # Не повторяем при других ошибках (например, 404)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                delay = backoff_factor * (2 ** i)
                print(f"Сетевая ошибка для {url}: {e}. Повторная попытка через {delay:.1f} сек. (Попытка {i + 1}/{retries})")
                await asyncio.sleep(delay)
            except Exception as e:
                print(f"Неожиданное исключение при загрузке {url}: {e}")
                return None # Не повторяем при неизвестных ошибках

    print(f"Не удалось загрузить {url} после {retries} попыток.")
    return None
    
async def asyncLoader(links):
    """
    Асинхронно загружает и обрабатывает HTML-страницы по ссылкам.
    Удаляет ненужные блоки и сохраняет очищенный текст в файлы.
    """
    if not links:
        print("Список ссылок пуст. Завершение работы.")
        return

    print(f"Загрузка {len(links)} ссылок...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    semaphore = asyncio.Semaphore(5)  # Не более 5 одновременных запросов

    # Создаем сессию aiohttp и асинхронно загружаем все страницы
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [_fetch_and_create_doc(session, link, semaphore) for link in links]
        # Используем tqdm_asyncio.gather для отображения прогресс-бара
        results = await tqdm_asyncio.gather(*tasks, desc="Загрузка страниц")
        docs = [doc for doc in results if doc is not None]
    print("Загрузка завершена. Начало обработки...")

    # Трансформер для удаления тегов по классам
    bs_transformer = BeautifulSoupTransformer()

    # Список классов для удаления. Добавлены новые на основе анализа.
    unwanted_classes = [
        # Стандартные блоки
        "header",
        "footer",
        "breadcrumbs",
        
        # Виджеты и боковые панели
        "p-category-filters",
        "b24-widget-button-wrapper",
        "blog-article-menu",
        
        # Рекламные и навигационные блоки
        "p-category-content-categories", # Блоки подкатегорий
        "p-category-content-products-title", # Заголовок "Найдено N товаров" и сортировка
        "main-top-block__info",
        "collections-slider", # Слайдер "Товарные коллекции"
        "clients-wrap", # Блок "Нам доверяют"
        "interst", # Блок со статьями "Статьи которые вас заинтересуют"
        "seen", # Блок "Вы смотрели"
        
        # Контентные блоки, которые не нужны на странице категории
        "p-category-content-products-items", # Сетка с карточками товаров
        "showmore", # Кнопка "Показать еще"
        "controls", # Пагинация и управление количеством товаров
        
        # Блоки из статей, которые могут быть на других страницах
        "blog-article-slider",
        "blog__subscribe",

        # Блоки со страницы товара для удаления
        "p-product-gallery-thumbs", # Миниатюры галереи изображений товара
        "p-product-gallery-wrap",   # Основная галерея изображений товара
        "p-product-content-products", # Блок "С этим товаром покупают"
        "user-photos-w",            # Блок с фотографиями покупателей
        "reviews",                  # Весь блок с отзывами
        "fancy-modal",
        "modal",
        "soglasie",
        "bottom-panel",
        "disabled-cart",
        "bx_breadcrumbs",
        "item-caption-title",
        "schema-block",
        "cheaper",
        "p-product-controls",
        "js-close-filter",
        "p-category-filters-title",
        "delivery-text"
    ]

    for doc in docs:
        # 1. Удаляем большие ненужные блоки с помощью bs_transformer
        cleaned_html = bs_transformer.remove_unwanted_classnames(
            doc.page_content, unwanted_classes
        )
        
        # 2. Используем BeautifulSoup для более тонкой обработки HTML
        soup = BeautifulSoup(cleaned_html, 'html.parser')
        
        # Находим все контейнеры с опциями товара (например, "Объем")
        option_containers = soup.find_all('div', class_='p-product-content-options')
        for container in option_containers:
            # Находим все теги 'a' внутри контейнера, которые являются опциями
            options = container.find_all('a', recursive=False)
            
            # Удаляем все опции, у которых нет класса 'active'
            for option in options:
                if 'class' not in option.attrs or 'active' not in option.attrs['class']:
                    option.decompose() # Метод decompose удаляет тег из дерева

        # Обновляем содержимое документа измененным HTML
        doc.page_content = str(soup)

    # Трансформер для преобразования HTML в текст
    html2text = Html2TextTransformer(ignore_links=True, ignore_images=True)
    docs_transformed = html2text.transform_documents(docs)



    for idx, doc in enumerate(docs_transformed):
        file_path = f"{DIR_TO_STORE}/document_{idx}.txt"
        try:
            with open(file_path, "w+", encoding="utf-8") as f:
                f.write(doc.page_content)
                print(f"Файл {file_path} сохранен")
        except Exception as e:
            print(f"Не удалось сохранить файл {file_path}: {e}")


if __name__ == "__main__":
    # Убедимся, что директория для сохранения документов существует
    if not os.path.exists(DIR_TO_STORE):
        os.makedirs(DIR_TO_STORE)

    links_to_parse = getLinks2Parse()
    # Запускаем асинхронную функцию asyncLoader
    asyncio.run(asyncLoader(links_to_parse))