

import time
import pymongo
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ============ 🔐 MongoDB CONFIG ============
MONGODB_URI = "mongodb+srv://hieuducle:255381228@chat-rag.qsimdqe.mongodb.net/?retryWrites=true&w=majority&appName=chat-rag"
DB_NAME = "chat-rag"
COLLECTION_NAME = "grocery"

# ============ 💡 Embedding Model ============
embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert")

def get_embedding(text: str) -> list[float]:
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# ============ Crawl using Selenium ============
def crawl_article(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        time.sleep(3)  

        soup = BeautifulSoup(driver.page_source, "html.parser")

        title = soup.find("div", class_="product-title").find("h1").text.strip()
        brand = soup.find("span", class_="product-type").find("b").text.strip()
        status = soup.find("span", class_="product-status").find("b").text.strip()

        price_tag = soup.select_one("#price-preview .pro-price")
        if price_tag:
            price = price_tag.text.strip()
            clean_price = int(price.replace(",", "").replace(".", "").replace("₫", "").strip())
            print(f"✅ Giá: {price} → {clean_price}")
        else:
            print("Không tìm thấy giá.")
            price, clean_price = "", 0

        # Mô tả sản phẩm
        desc_div = soup.find("div", class_="product-short-desc")
        description = ""
        if desc_div:
            li_items = desc_div.find_all("li")
            if li_items:
                description = "\n".join(li.get_text(strip=True) for li in li_items)
            else:
                span = desc_div.find("span")
                if span and "<br" in str(span):
                    lines = str(span).split("<br>")
                    description = "\n".join(BeautifulSoup(line, "html.parser").get_text(strip=True) for line in lines)
                else:
                    p_tags = desc_div.find_all("p")
                    description = "\n".join(p.get_text(strip=True) for p in p_tags if p.get_text(strip=True))

        # Ảnh sản phẩm
        img_tag = soup.find("img", class_="product-image-feature")
        if img_tag:
            if img_tag["src"].startswith("//"):
                img_url = "https:" + img_tag["src"]
            else:
                img_url = img_tag["src"]
        else:
            img_url = ""

        return {
            "name_title": title,
            "image": img_url,
            "status": status,
            "brand": brand,
            "description": description,
            "source_url": url,
            "price_display": price,
            "price_value": clean_price,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        print(f"Lỗi khi crawl {url}: {e}")
        return None

    finally:
        driver.quit()

def save_to_mongodb(data_list):
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print(" Kết nối MongoDB thành công")

        if data_list:
            collection.insert_many(data_list)
            print(f"Đã lưu {len(data_list)} sản phẩm vào MongoDB.")
        else:
            print("Không có dữ liệu để lưu.")
    except Exception as e:
        print("Lỗi khi lưu vào MongoDB:", e)


if __name__ == "__main__":
    # all_data = []
    # try:
    #     with open("urls.txt", "r", encoding="utf-8") as f:
    #         urls = [line.strip() for line in f if line.strip()]
    # except FileNotFoundError:
    #     print("❌ File urls.txt không tồn tại.")
    #     urls = []

    # for i, url in enumerate(urls, 1):
    #     print(f"\n🔍 [{i}/{len(urls)}] Đang xử lý: {url}")
    #     result = crawl_article(url)
    #     if result:
    #         all_data.append(result)

    # if all_data:
    #     save_to_mongodb(all_data)
    # else:
    #     print("Không có dữ liệu nào được crawl.")


    #  Connect to MongoDB
    client = pymongo.MongoClient(MONGODB_URI) 
    db = client[DB_NAME]  
    collection = db["grocery"] 
    data = list(collection.find())

   
    df = pd.DataFrame(data)
    df["embedding"] = df["name_title"].apply(get_embedding)
    collection = db['embedding_for_vector_search_new']
    collection.delete_many({})
    documents = df.to_dict("records")
    collection.insert_many(documents)

    print("Data ingestion into MongoDB completed")


