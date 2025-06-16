from flask import Flask, request, jsonify
from flask_cors import CORS
import pymongo
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import json

from semantic_router.samples import productsSample, chitchatSample, policiesSample
from reflection import Reflection
from semantic_router import SemanticRouter, Route
from embeddings.sentenceTransformer import SentenceTransformerEmbedding
from embeddings.base import BaseEmbedding, EmbeddingConfig

import chromadb
from bson import ObjectId

load_dotenv()


app = Flask(__name__)
CORS(app)


MONGODB_URI = os.getenv("MONGODB_URI")
cluster = pymongo.MongoClient(MONGODB_URI)
db = cluster["chat-rag"]
collection = db["embedding_for_vector_search_new"]

embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert")


client = chromadb.PersistentClient(path="./rag_chatbot_data")
config = EmbeddingConfig(name="keepitreal/vietnamese-sbert")
embedding_function = SentenceTransformerEmbedding(config)

try:
    collection_chromadb = client.get_collection(name="policies_order", embedding_function=embedding_function)
    print("Đã kết nối với collection_chromadb: policies_order")
except:
    raise Exception("Collection_chromadb 'policies_order' không tồn tại.")


chat_history = []

class DatabaseAnalyzer:
    """Xử lý các câu hỏi phân tích database"""
    
    def __init__(self, collection, llm_client):
        self.collection = collection
        self.llm_client = llm_client
    
    def serialize_mongodb_result(self, obj):
        """Chuyển đổi ObjectId và các kiểu MongoDB khác thành JSON serializable"""
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self.serialize_mongodb_result(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.serialize_mongodb_result(item) for item in obj]
        else:
            return obj
        
    def is_analytics_query(self, query):
        """Kiểm tra xem câu hỏi có phải là analytics query không"""
        analytics_keywords = [
            'bao nhiêu', 'có mấy', 'đếm', 'tổng cộng', 'thống kê',
            'phân tích', 'so sánh', 'nhiều nhất', 'ít nhất', 'trung bình',
            'dưới', 'trên', 'từ', 'đến', 'giữa', 'khoảng'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in analytics_keywords)
    
    def extract_conditions_with_llm(self, query):
        """Sử dụng LLM để trích xuất điều kiện từ câu hỏi"""
        
        system_prompt = """
        Bạn là một chuyên gia phân tích câu hỏi để trích xuất thông tin database.
        
        Nhiệm vụ: Phân tích câu hỏi và trả về JSON với các thông tin sau:
        {
            "action": "count|sum|avg|max|min|find",
            "filters": {
                "name_title": "từ khóa tìm kiếm trong tên sản phẩm",
                "brand": "thương hiệu",
                "price_min": số_min,
                "price_max": số_max,
                "category": "danh mục sản phẩm"
            },
            "target_field": "trường cần thống kê (price_display, name_title, etc.)"
        }
        
        QUAN TRỌNG - Quy tắc chuyển đổi tiền tệ:
        - "nghìn" = x1000 (ví dụ: 100 nghìn = 100000)
        - "ngàn" = x1000 (ví dụ: 50 ngàn = 50000)  
        - "k" = x1000 (ví dụ: 20k = 20000)
        - "triệu" = x1000000 (ví dụ: 1 triệu = 1000000)
        - Số không có đơn vị = giữ nguyên (ví dụ: 50000 = 50000)
        
        Ví dụ:
        - "có bao nhiêu sản phẩm sữa dưới 30 nghìn" 
        -> {"action": "count", "filters": {"name_title": "sữa", "price_max": 30000}}
        
        - "có bao nhiêu sản phẩm trong cửa hàng"
        -> {"action": "count", "filters": {}}
        
        - "có những sản phẩm nào dưới 100 nghìn"
        -> {"action": "find", "filters": {"price_max": 100000}}
        
        - "sản phẩm từ 50k đến 200 nghìn"
        -> {"action": "find", "filters": {"price_min": 50000, "price_max": 200000}}
        
        - "có mấy sản phẩm trên 1 triệu"
        -> {"action": "count", "filters": {"price_min": 1000000}}
        
        Chỉ trả về JSON, không giải thích gì thêm.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.llm_client.chat(messages)
            
            json_str = response.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:-3]
            elif json_str.startswith('```'):
                json_str = json_str[3:-3]
            
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {"action": "count", "filters": {}}
    
    def build_mongodb_query(self, conditions):
        """Xây dựng MongoDB query từ điều kiện"""
        query = {}
        
        filters = conditions.get("filters", {})
        
     
        if "name_title" in filters and filters["name_title"]:
            query["name_title"] = {"$regex": filters["name_title"], "$options": "i"}
        
    
        if "brand" in filters and filters["brand"]:
            query["brand"] = {"$regex": filters["brand"], "$options": "i"}
        
      
        price_filter = {}
        if "price_min" in filters and filters["price_min"]:
            price_filter["$gte"] = filters["price_min"]
        if "price_max" in filters and filters["price_max"]:
            price_filter["$lte"] = filters["price_max"]
        
        if price_filter:
          
            query["price_value"] = price_filter
        
        return query
    
    def execute_analytics_query(self, conditions):
        """Thực thi câu hỏi phân tích"""
        action = conditions.get("action", "count")
        query = self.build_mongodb_query(conditions)
        
        print(f"MongoDB Query: {query}")
        
        try:
            if action == "count":
                result = self.collection.count_documents(query)
                return {"type": "count", "value": result, "query": query}
            
            elif action == "find":
                
                projection = {
                    "_id": 0,  
                    "name_title": 1,
                    "image": 1,
                    "status": 1,
                    "brand": 1,
                    "price_display": 1,
                    "price_value": 1,
                    "description": 1
                }
                
                results = list(self.collection.find(query, projection).limit(10))
                
                serialized_results = self.serialize_mongodb_result(results)
                return {"type": "find", "value": serialized_results, "count": len(results), "query": query}
            
            elif action in ["sum", "avg", "max", "min"]:
                target_field = conditions.get("target_field", "price_value")
                
                pipeline = [
                    {"$match": query}
                ]
                
                
                if action == "sum":
                    pipeline.append({"$group": {"_id": None, "total": {"$sum": f"${target_field}"}}})
                elif action == "avg":
                    pipeline.append({"$group": {"_id": None, "average": {"$avg": f"${target_field}"}}})
                elif action == "max":
                    pipeline.append({"$group": {"_id": None, "maximum": {"$max": f"${target_field}"}}})
                elif action == "min":
                    pipeline.append({"$group": {"_id": None, "minimum": {"$min": f"${target_field}"}}})
                
                results = list(self.collection.aggregate(pipeline))
            
                serialized_results = self.serialize_mongodb_result(results)
                return {"type": action, "value": serialized_results[0] if serialized_results else None, "query": query}
            
        except Exception as e:
            print(f"Error executing analytics query: {e}")
            return {"type": "error", "value": str(e), "query": query}
    
    def format_analytics_response(self, query, result):
        """Định dạng kết quả phân tích thành câu trả lời tự nhiên"""
        
        system_prompt = f"""
        Bạn là nhân viên cửa hàng tạp hóa thân thiện. 
        Khách hàng vừa hỏi: "{query}"
        
        Kết quả từ database: {json.dumps(result, ensure_ascii=False)}
        
        Hãy trả lời một cách tự nhiên và thân thiện, bao gồm:
        1. Trả lời trực tiếp câu hỏi
        2. Đưa ra thông tin bổ sung nếu hữu ích
        3. Gợi ý hoặc khuyến khích nếu phù hợp
        
        Nếu action là "find" và có danh sách sản phẩm, hãy liệt kê tên và giá của từng sản phẩm.
        
        Ví dụ:
        - Nếu có 15 sản phẩm sữa dưới 30k: "Cửa hàng chúng tôi hiện có 15 sản phẩm sữa với giá dưới 30.000đ. Bạn có muốn tôi gợi ý một số sản phẩm phù hợp không?"
        - Nếu tìm thấy danh sách sản phẩm: "Tôi tìm thấy các sản phẩm sau đây dưới 100.000đ: [liệt kê tên và giá]"
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Hãy trả lời dựa trên kết quả: {result}"}
        ]
        
        return self.llm_client.chat(messages)

class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def chat(self, messages):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return completion.choices[0].message.content

def get_embedding(text):
    if not text.strip():
        return []
    embedding = embedding_model.encode(text)
    return embedding.tolist()

def vector_search(user_query, collection, limit=4):
    query_embedding = get_embedding(user_query)
    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 320,
            "limit": limit,
        }
    }

    unset_stage = {"$unset": "embedding"}
    project_stage = {
        "$project": {
            "_id": 0,
            "name_title": 1,
            "image": 1,
            "status": 1,
            "brand": 1,
            "price_display": 1,
            "price_value": 1,  
            "description": 1
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]
    results = collection.aggregate(pipeline)
    return list(results)

def get_search_result(query, collection):
    get_knowledge = vector_search(query, collection, 4)
    search_result = ""
    i = 0
    for result in get_knowledge:
        i += 1
        search_result += f"\n {i}"
        if result.get("name_title"):
            search_result += f" Tên sản phẩm: {result.get('name_title')}"
        if result.get("image"):  
            search_result += f" Hình ảnh: {result.get('image')}"
        if result.get("price_display"):
            search_result += f" Giá sản phẩm: {result.get('price_display')}"
        if result.get("status"):
            search_result += f" Tình trạng: {result.get('status')}"
        if result.get("brand"):
            search_result += f" Thương hiệu: {result.get('brand')}"
        if result.get("description"):
            search_result += f" Miêu tả: {result.get('description')}"
    return search_result

def handle_query(query: str, collection_chromadb):
    results = collection_chromadb.query(query_texts=[query], n_results=3)
    return "\n".join(results['documents'][0])

# Initialize components
config = BaseEmbedding("keepitreal/vietnamese-sbert")
transfromerEmbedding = SentenceTransformerEmbedding(config)

PRODUCT_ROUTE_NAME = 'products'
CHITCHAT_ROUTE_NAME = 'chitchat'
POLICIES_ROUTE_NAME = 'policies'
ANALYTICS_ROUTE_NAME = 'analytics' 


analyticsSample = [
    "có bao nhiêu sản phẩm trong cửa hàng",
    "có mấy sản phẩm sữa",
    "đếm số sản phẩm dưới 30 nghìn",
    "có bao nhiêu sản phẩm sữa dưới 30 nghìn",
    "thống kê sản phẩm theo giá",
    "sản phẩm nào đắt nhất",
    "tổng cộng có mấy loại bánh",
    "có bao nhiêu sản phẩm của thương hiệu vinamilk"
]

productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
policiesRoute = Route(name=POLICIES_ROUTE_NAME, samples=policiesSample)
analyticsRoute = Route(name=ANALYTICS_ROUTE_NAME, samples=analyticsSample)

semanticRouter = SemanticRouter(transfromerEmbedding, routes=[productRoute, chitchatRoute, policiesRoute, analyticsRoute])


openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai_client = OpenAIClient(api_key=openai_api_key)
reflection = Reflection(llm=openai_client)


db_analyzer = DatabaseAnalyzer(collection, openai_client)

@app.route('/')
def root():
    return jsonify({"message": "Enhanced Chat API with Database Analytics is running"})

@app.route('/chat', methods=['POST'])
def ask_product_question():
    global chat_history
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid request format"}), 400
            
        if isinstance(data['message'], dict) and 'content' in data['message']:
            query = data['message']['content'].strip()
        elif isinstance(data['message'], str):
            query = data['message'].strip()
        else:
            return jsonify({"error": "Invalid message format"}), 400
            
        if not query:
            return jsonify({"error": "Câu hỏi không hợp lệ."}), 400

    
        guidedRoute = semanticRouter.guide(query)[1]
        print(f"Guided Route: {guidedRoute}")
        
      
        if guidedRoute == ANALYTICS_ROUTE_NAME:
            print("Guide to Database Analytics")
            
            chat_history.append({"role": "user", "content": query})
            
            
            conditions = db_analyzer.extract_conditions_with_llm(query)
            print(f"Extracted conditions: {conditions}")
            
         
            result = db_analyzer.execute_analytics_query(conditions)
            print(f"Analytics result: {result}")
            
           
            response = db_analyzer.format_analytics_response(query, result)
            print(f"AI: {response}")
            
            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"content": response, "analytics_data": result})
        
        
        elif guidedRoute == PRODUCT_ROUTE_NAME:
            print("Guide to RAGs Product")
            
            chat_history.append({"role": "user", "content": query})
            reflected_query = reflection(chat_history)
            print("Reflected Query:", reflected_query)

            source_information = get_search_result(reflected_query, collection).replace('<br>', '\n')

            prompt = (
                f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng tạp hóa.\n"
                f"Câu hỏi của khách hàng: {reflected_query}\n"
                f"Trả lời câu hỏi dựa vào các thông tin sản phẩm sau:\n{source_information}, đây là câu hỏi chỉ dành cho cửa hàng của mình nên hãy trả lời chỉ nằm trong thông tin được cung cấp, nếu không có thông tin thì hãy trả lời là không có, chứ đừng tạo thông tin giả"
                f"Nếu như khách hàng hỏi về sản phẩm cụ thể trong cửa hàng, nếu có thông tin thì hãy trả lời đầy đủ về tên sản phẩm, giá sản phẩm, và cả hình ảnh sản phẩm."
            )

            chat_history.append({"role": "system", "content": prompt})
            chat_history.append({"role": "user", "content": reflected_query})

            response = openai_client.chat(chat_history)
            print(f"AI: {response}")

            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"content": response})

        elif guidedRoute == POLICIES_ROUTE_NAME:
            print("Guide to RAG Policies")
            
            chat_history.append({"role": "user", "content": query})
            reflected_query = reflection(chat_history)
            print("Reflected Query:", reflected_query)

            source_information = handle_query(reflected_query, collection_chromadb)

            prompt = (
                f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng tạp hóa.\n"
                f"Câu hỏi của khách hàng: {reflected_query}, đây là câu hỏi chỉ dành cho cửa hàng của mình nên hãy trả lời chỉ nằm trong thông tin được cung cấp \n"
                f"Trả lời câu hỏi dựa vào các thông tin sau:\n{source_information}"
            )

            chat_history.append({"role": "system", "content": prompt})
            chat_history.append({"role": "user", "content": reflected_query})

            response = openai_client.chat(chat_history)
            print(f"AI: {response}")

            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"content": response})

        elif guidedRoute == CHITCHAT_ROUTE_NAME:
            print("Guide to Chitchat")
            
            if not chat_history:
                chat_history.append({
                    "role": "system",
                    "content": "Bạn là một nhân viên thân thiện tại cửa hàng tạp hóa. Trả lời tự nhiên và dễ gần."
                })

            chat_history.append({"role": "user", "content": query})
            response = openai_client.chat(chat_history)
            print(f"AI: {response}")
            
            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"content": response})
        
        else:
         
            if not chat_history:
                chat_history.append({
                    "role": "system",
                    "content": "Bạn là một nhân viên thân thiện tại cửa hàng tạp hóa. Trả lời tự nhiên và dễ gần."
                })
            
            chat_history.append({"role": "user", "content": query})
            response = openai_client.chat(chat_history)
            chat_history.append({"role": "assistant", "content": response})
            
            return jsonify({"content": response})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    try:
        print("Starting Enhanced Server with Database Analytics...")
        app.run(host="0.0.0.0", port=8000, debug=True)
    except Exception as e:
        print(f"Failed to start server: {e}")