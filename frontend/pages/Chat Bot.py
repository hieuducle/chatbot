

import streamlit as st
import requests
import re
import uuid

# Set up the Streamlit interface
page = st.title("Assisttant Chatbot")

def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]

# Generate a random session ID
session_id = str(uuid.uuid4())

# Initialize the session state for the backend URL
if "flask_api_url" not in st.session_state:
    print('-go 1')
    st.session_state.flask_api_url = None

# Function to display the dialog and set the URL
@st.dialog("Setup Back end")
def vote():
    clear_session_state()
    st.markdown(
        """
        Run the backend [here](https://colab.research.google.com/drive/1XIZikuY3KtZzfnDG3wEcBBIHJhhdKgU-?usp=sharing) and paste the Ngrok link below.
        """
    )
    link = st.text_input("Backend URL", "")
    if st.button("Save"):
        st.session_state.flask_api_url = "{}/chat".format(link)
        st.rerun()

# Display the dialog only if the URL is not set
if st.session_state.flask_api_url is None:
    print('-go 2')
    vote()

# Once the URL is set, display it or proceed with other functionality
if "flask_api_url" in st.session_state:
    st.write(f"Backend is set to: {st.session_state.flask_api_url}")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def extract_and_display_images(text):
    """
    Tìm và hiển thị hình ảnh từ URL trong text
    """
    # Pattern để tìm URL hình ảnh
    image_url_pattern = r'https?://[^\s<>"]+?\.(?:jpg|jpeg|png|gif|webp|bmp)'
    
    # Tìm tất cả URL hình ảnh
    image_urls = re.findall(image_url_pattern, text, re.IGNORECASE)
    
    if image_urls:
        # Hiển thị text trước
        st.markdown(text)
        
        # Hiển thị từng hình ảnh
        for i, url in enumerate(image_urls):
            try:
                st.image(url, caption=f"Hình ảnh sản phẩm {i+1}", width=300)
            except Exception as e:
                st.error(f"Không thể tải hình ảnh: {url}")
                st.write(f"URL: {url}")
    else:
        # Nếu không có hình ảnh, chỉ hiển thị text
        st.markdown(text)

def format_response_with_images(response_text):
    """
    Format response để hiển thị hình ảnh một cách đẹp mắt
    """
    # Pattern để tìm URL hình ảnh kèm theo context
    image_pattern = r'(Hình ảnh:\s*)(https?://[^\s<>"]+?\.(?:jpg|jpeg|png|gif|webp|bmp))'
    
    # Thay thế URL hình ảnh bằng markdown image syntax
    formatted_text = re.sub(image_pattern, r'\1\n\n![](\2)\n\n', response_text, flags=re.IGNORECASE)
    
    return formatted_text

# Display the chat history using chat UI
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Kiểm tra và hiển thị hình ảnh cho response của assistant
            extract_and_display_images(message["content"])
        else:
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Prepare the payload for the request
        payload = {
            "message": {"content": prompt},
            "sessionId": session_id
        }
        
        try:
            # Send the POST request to the Flask API
            response = requests.post(st.session_state.flask_api_url, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                # Get the response from the API
                api_response = response.json()
                response_content = api_response['content']
                
                # Hiển thị response với hình ảnh
                extract_and_display_images(response_content)
                
                # Add the assistant's response to the chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response_content})
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# Thêm custom CSS để styling hình ảnh
st.markdown("""
<style>
.stImage > div {
    display: flex;
    justify-content: center;
}

.stImage img {
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin: 10px 0;
}

/* Custom styling for chat messages with images */
.stChatMessage [data-testid="stImage"] {
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)