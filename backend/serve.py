from flask import Flask, request, jsonify
from flask_cors import CORS
import pymongo
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

from semantic_router.samples import productsSample, chitchatSample, policiesSample
from reflection import Reflection
from semantic_router import SemanticRouter, Route
from embeddings.sentenceTransformer import SentenceTransformerEmbedding
from embeddings.base import BaseEmbedding, EmbeddingConfig

import chromadb

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
MONGODB_URI = os.getenv("MONGODB_URI")
cluster = pymongo.MongoClient(MONGODB_URI)
db = cluster["chat-rag"]
collection = db["embedding_for_vector_search_new"]

embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert")




# Khởi tạo ChromaDB client
client = chromadb.PersistentClient(path="./backend/rag_chatbot_data")

# Khởi tạo embedding function (phải giống với lúc tạo collection)
config = EmbeddingConfig(name="keepitreal/vietnamese-sbert")
embedding_function = SentenceTransformerEmbedding(config)

# Lấy collection hiện có
try:
    collection_chromadb = client.get_collection(name="policies_order",embedding_function=embedding_function)
    print("Đã kết nối với collection_chromadb: policies_order")
except:
    raise Exception("Collection_chromadb 'policies_order' không tồn tại. Vui lòng kiểm tra lại database hoặc tạo collection_chromadb mới.")

# Hàm xử lý truy vấn
def handle_query(query: str, collection_chromadb):
    results = collection_chromadb.query(query_texts=[query], n_results=3)
    return "\n".join(results['documents'][0])





# Global chat history (in-memory storage)
chat_history = []

class OpenAIClient():
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
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)
    return embedding.tolist()

def vector_search(user_query, collection, limit=4):
    """
    Perform a vector search in the MongoDB collection based on the user query.
    """
    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 320,
            "limit": limit,
        }
    }

    unset_stage = {
        "$unset": "embedding"
    }

    project_stage = {
        "$project": {
            "_id": 0,
            "name_title": 1,
            "image": 1,
            "status": 1,
            "brand": 1,
            "price_display": 1,
            "description": 1
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

def get_search_result(query, collection):
    get_knowledge = vector_search(query, collection, 4)

    search_result = ""
    i = 0
    for result in get_knowledge:
        i += 1
        search_result += "\n {}".format(i)
        if result.get("name_title"):
            search_result += " Tên sản phẩm: {}".format(result.get("name_title"))
        if result.get("image"):  
            search_result += " Hình ảnh: {}".format(result.get("image"))
        if result.get("price_display"):
            search_result += " Giá sản phẩm: {}".format(result.get("price_display"))
        if result.get("status"):
            search_result += " Tình trạng: {}".format(result.get("status"))
        if result.get("brand"):
            search_result += " Thương hiệu: {}".format(result.get("brand"))
        if result.get("description"):
            search_result += " Miêu tả: {}".format(result.get("description"))
    return search_result

# Initialize components
config = BaseEmbedding("keepitreal/vietnamese-sbert")
transfromerEmbedding = SentenceTransformerEmbedding(config)

PRODUCT_ROUTE_NAME = 'products'
CHITCHAT_ROUTE_NAME = 'chitchat'
POLICIES_ROUTE_NAME = 'policies'
productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
policiesRoute = Route(name=POLICIES_ROUTE_NAME, samples=policiesSample)

semanticRouter = SemanticRouter(transfromerEmbedding, routes=[productRoute, chitchatRoute,policiesRoute])

# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai_client = OpenAIClient(api_key=openai_api_key)
reflection = Reflection(llm=openai_client)

@app.route('/')
def root():
    return jsonify({"message": "Chat API is running"})

@app.route('/chat', methods=['POST'])
def ask_product_question():
    global chat_history
    try:
        # Get request data
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid request format"}), 400
            
        # Extract message content
        if isinstance(data['message'], dict) and 'content' in data['message']:
            query = data['message']['content'].strip()
        elif isinstance(data['message'], str):
            query = data['message'].strip()
        else:
            return jsonify({"error": "Invalid message format"}), 400
            
        if not query:
            return jsonify({"error": "Câu hỏi không hợp lệ."}), 400

        # Guide the route using semantic router
        guidedRoute = semanticRouter.guide(query)[1]
        print(f"Guided Route: {guidedRoute}")
        
        if guidedRoute == PRODUCT_ROUTE_NAME:
            print("Guide to RAGs Product")
            
            # Add user message to chat history
            chat_history.append({"role": "user", "content": query})
            
            # Use reflection to improve query
            reflected_query = reflection(chat_history)
            print("Reflected Query:", reflected_query)

            # Get search resultsS
            source_information = get_search_result(reflected_query, collection).replace('<br>', '\n')

            # Create prompt
            prompt = (
                f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng tạp hóa.\n"
                f"Câu hỏi của khách hàng: {reflected_query}\n"
                f"Trả lời câu hỏi dựa vào các thông tin sản phẩm sau:\n{source_information}, đây là câu hỏi chỉ dành cho cửa hàng của mình nên hãy trả lời chỉ nằm trong thông tin được cung cấp, nếu không có thông tin thì hãy trả lời là không có, chứ đừng tạo thông tin giả"
                f"Nếu như khách hàng hỏi về sản phẩm cụ thể trong cửa hàng, nếu có thông tin thì hãy trả lời đầy đủ về tên sản phẩm, giá sản phẩm, và cả hình ảnh sản phẩm."
            )

            # Add system prompt and reflected query to chat history
            chat_history.append({"role": "system", "content": prompt})
            chat_history.append({"role": "user", "content": reflected_query})

            # Get response from OpenAI
            response = openai_client.chat(chat_history)
            print(f"AI: {response}")

            # Add response to chat history
            chat_history.append({"role": "assistant", "content": response})
            
            return jsonify({"content": response})

        elif guidedRoute == POLICIES_ROUTE_NAME:
            print("Guide to RAG Policies")
            
            # Add user message to chat history
            chat_history.append({"role": "user", "content": query})
            
            # Use reflection to improve query
            reflected_query = reflection(chat_history)
            print("Reflected Query:", reflected_query)

            # Get search resultsS
            source_information = handle_query(reflected_query, collection_chromadb)
            

            # Create prompt
            prompt = (
                f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng tạp hóa.\n"
                f"Câu hỏi của khách hàng: {reflected_query}, đây là câu hỏi chỉ dành cho cửa hàng của mình nên hãy trả lời chỉ nằm trong thông tin được cung cấp \n"
                f"Trả lời câu hỏi dựa vào các thông tin sau:\n{source_information}"
            )

            # Add system prompt and reflected query to chat history
            chat_history.append({"role": "system", "content": prompt})
            chat_history.append({"role": "user", "content": reflected_query})

            # Get response from OpenAI
            response = openai_client.chat(chat_history)
            print(f"AI: {response}")

            # Add response to chat history
            chat_history.append({"role": "assistant", "content": response})
            
            return jsonify({"content": response})

        elif guidedRoute == CHITCHAT_ROUTE_NAME:
            print("Guide to Chitchat")
            
            # Add system message if chat history is empty
            if not chat_history:
                chat_history.append({
                    "role": "system",
                    "content": "Bạn là một nhân viên thân thiện tại cửa hàng tạp hóa. Trả lời tự nhiên và dễ gần."
                })

            # Add user message to chat history
            chat_history.append({"role": "user", "content": query})
            
            # Get response from OpenAI
            response = openai_client.chat(chat_history)
            print(f"AI: {response}")
            
            # Add response to chat history
            chat_history.append({"role": "assistant", "content": response})
            
            return jsonify({"content": response})
        
        else:
            # Default fallback
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
        print("Starting server...")
        app.run(host="0.0.0.0", port=8000, debug=True)
    except Exception as e:
        print(f"Failed to start server: {e}")