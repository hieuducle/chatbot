
import streamlit as st
import requests
import re
import uuid

# Set up the Streamlit interface
page = st.title("Grocery Chatbot")

def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]


session_id = str(uuid.uuid4())


if "flask_api_url" not in st.session_state:
    print('-go 1')
    st.session_state.flask_api_url = None


@st.dialog("Setup Back end")
def vote():
    clear_session_state()
    st.markdown(
        """
        ⚙️ **Paste your backend URL below.**  
        Example: `http://127.0.0.1:5000` 
        """
    )
    link = st.text_input("Backend URL", "")
    if st.button("Save"):
        st.session_state.flask_api_url = "{}/chat".format(link)
        st.rerun()


if st.session_state.flask_api_url is None:
    print('-go 2')
    vote()


if "flask_api_url" in st.session_state:
    st.write(f"Backend is set to: {st.session_state.flask_api_url}")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def extract_and_display_images(text):
    """
    Tìm và hiển thị hình ảnh từ URL trong text
    """
  
    image_url_pattern = r'https?://[^\s<>"]+?\.(?:jpg|jpeg|png|gif|webp|bmp)'
    
   
    image_urls = re.findall(image_url_pattern, text, re.IGNORECASE)
    
    if image_urls:
        
        st.markdown(text)
        
    
        for i, url in enumerate(image_urls):
            try:
                st.image(url, caption=f"Hình ảnh sản phẩm {i+1}", width=300)
            except Exception as e:
                st.error(f"Không thể tải hình ảnh: {url}")
                st.write(f"URL: {url}")
    else:
    
        st.markdown(text)

def format_response_with_images(response_text):
 
  
    image_pattern = r'(Hình ảnh:\s*)(https?://[^\s<>"]+?\.(?:jpg|jpeg|png|gif|webp|bmp))'
    

    formatted_text = re.sub(image_pattern, r'\1\n\n![](\2)\n\n', response_text, flags=re.IGNORECASE)
    
    return formatted_text


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
           
            extract_and_display_images(message["content"])
        else:
            st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    
    st.session_state.chat_history.append({"role": "user", "content": prompt})
 
    with st.chat_message("user"):
        st.markdown(prompt)
    
   
    with st.chat_message("assistant"):
    
        payload = {
            "message": {"content": prompt},
            "sessionId": session_id
        }
        
        try:
          
            response = requests.post(st.session_state.flask_api_url, json=payload)

           
            if response.status_code == 200:
                
                api_response = response.json()
                response_content = api_response['content']
                
            
                extract_and_display_images(response_content)
                
             
                st.session_state.chat_history.append({"role": "assistant", "content": response_content})
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")


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