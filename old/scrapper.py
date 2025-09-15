import os
import asyncio
import aiohttp
import json
import random
from tqdm.asyncio import tqdm_asyncio
from typing import List, Dict, Any, Optional

# --- КОНФИГУРАЦИЯ ---
# Базовый URL вашего API
BASE_URL = "https://opt6.ru/local/api/export_products.php" 
# ВАЖНО: Замените 'YOUR_SECRET_KEY' на ваш реальный секретный ключ из export_products.php
SECRET_KEY = "A4AE72FAF451585C45E27598B2FEF" 
# Директория для сохранения текстовых файлов с данными о товарах
DIR_TO_STORE = "docs"
# Количество одновременных запросов к API. Снижает риск блокировки.
CONCURRENT_REQUESTS = 5
# --- КОНЕЦ КОНФИГУРАЦИИ ---

def format_product_to_text(product: Dict[str, Any]) -> str:
    """
    Форматирует словарь с данными о товаре в читаемый текстовый формат.
    """
    content_parts = []

    content_parts.append(f"Название товара: {product.get('name', 'Не указано')}")
    content_parts.append(f"ID товара: {product.get('id', 'Не указано')}")
    content_parts.append(f"URL: {product.get('url', 'Не указано')}")

    categories = product.get('categories', [])
    if categories:
        cat_names = [cat.get('name', '') for cat in categories]
        content_parts.append(f"Категории: {', '.join(cat_names)}")

    content_parts.append(f"Количество на складе: {product.get('stock_quantity', 0)}")
    
    content_parts.append("\nОписание:")
    content_parts.append(product.get('description', 'Нет описания.'))

    prices = product.get('prices', [])
    if prices:
        content_parts.append("\nЦены:")
        for price_tier in prices:
            qty_from = price_tier.get('quantity_from', 1)
            qty_to = price_tier.get('quantity_to')
            price = price_tier.get('price', 'N/A')
            if qty_to:
                content_parts.append(f"- От {qty_from} до {qty_to} шт.: {price} руб.")
            else:
                content_parts.append(f"- От {qty_from} шт.: {price} руб.")

    properties = product.get('properties', {})
    if properties:
        content_parts.append("\nХарактеристики:")
        for prop_code, prop_data in properties.items():
            prop_name = prop_data.get('name', prop_code)
            prop_value = prop_data.get('value', 'Не указано')
            if isinstance(prop_value, list):
                prop_value = ", ".join(map(str, prop_value))
            content_parts.append(f"- {prop_name}: {prop_value}")
    
    return "\n".join(content_parts)

async def fetch_page_data(session: aiohttp.ClientSession, page: int, semaphore: asyncio.Semaphore) -> Optional[List[Dict]]:
    """
    Асинхронно загружает и парсит JSON с одной страницы API, используя семафор для ограничения конкурентности.
    """
    params = {'key': SECRET_KEY, 'page': page}
    async with semaphore:
        # Добавляем небольшую случайную задержку перед каждым запросом
        await asyncio.sleep(random.uniform(0.2, 1.0))
        try:
            async with session.get(BASE_URL, params=params, ssl=False, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('products', [])
                else:
                    print(f"Ошибка при загрузке страницы {page}: статус {response.status}")
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            print(f"Ошибка сети или парсинга для страницы {page}: {e}")
            return None

async def main():

    if not os.path.exists(DIR_TO_STORE):
        os.makedirs(DIR_TO_STORE)
        print(f"Создана директория: {DIR_TO_STORE}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(headers=headers) as session:
        # 1. Получаем общее количество страниц
        print("Получение информации о количестве страниц...")
        params_first_page = {'key': SECRET_KEY, 'page': 1}
        total_pages = 0
        try:
            # Первый запрос делаем без семафора, чтобы быстро получить метаданные
            async with session.get(BASE_URL, params=params_first_page, ssl=False, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    total_pages = data.get('pagination', {}).get('totalPages', 0)
                    print(f"Найдено страниц: {total_pages}")
                else:
                    print(f"Не удалось получить информацию о страницах. Статус: {response.status}")
                    return
        except Exception as e:
            print(f"Критическая ошибка при первом запросе: {e}")
            return
            
        if total_pages == 0:
            print("Не найдено товаров для обработки.")
            return

        # 2. Создаем задачи для всех страниц
        tasks = [fetch_page_data(session, page, semaphore) for page in range(1, total_pages + 1)]

        # 3. Асинхронно выполняем все задачи с прогресс-баром
        all_products = []
        
        results = await tqdm_asyncio.gather(*tasks, desc="Загрузка данных о товарах")
        for product_list in results:
            if product_list:
                all_products.extend(product_list)
        
        print(f"\nЗагрузка завершена. Всего получено товаров: {len(all_products)}. Начинается сохранение файлов...")

        # 4. Сохраняем каждый товар в отдельный файл
        for product in all_products:
            product_id = product.get('id')
            if not product_id:
                continue
            
            file_content = format_product_to_text(product)
            file_path = os.path.join(DIR_TO_STORE, f"product_{product_id}.txt")
            
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
            except IOError as e:
                print(f"Не удалось сохранить файл {file_path}: {e}")

    print(f"\nРабота завершена. Все товары сохранены в директории '{DIR_TO_STORE}'.")

if __name__ == "__main__":
    asyncio.run(main())

