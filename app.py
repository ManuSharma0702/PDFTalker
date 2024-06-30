import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from database import split_documents, add_to_chroma
from query import query_rag
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
import os
st.set_page_config(page_title="Talking with pdf",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Talk with your pdf 🤖")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
# Create a directory to store the PDF files
data_dir = "/home/manusharma/llama_model/rag_from_pdfs/data"

# Save the uploaded PDF file to the data directory
if uploaded_file is not None:
    pdf_path = os.path.join(data_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
document_loader = PyPDFDirectoryLoader("data")
documents = document_loader.load()
chunks = split_documents(documents)
add_to_chroma(chunks)

from dataclasses import dataclass
from typing import Literal
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import os
import streamlit.components.v1 as components

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["You", "PDF"]
    message: str

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation" not in st.session_state:
        llm = Ollama(model = "llama3")    
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm),
        )

def on_click_callback():
    
    human_prompt = st.session_state.human_prompt
    
    llm_response = query_rag(human_prompt)
   
  
    st.session_state.history.append(
        Message("You", human_prompt)
    )
    st.session_state.history.append(
        Message("PDF", llm_response)
    )
import base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

initialize_session_state()



chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()
ai_icon_base64 = get_base64_image("/home/manusharma/llama_model/rag_from_pdfs/static/ai_icon.png")
user_icon_base64 = get_base64_image("/home/manusharma/llama_model/rag_from_pdfs/static/user_icon.png")
with chat_placeholder:
    for chat in st.session_state.history:
        icon_base64 = ai_icon_base64 if chat.origin == 'PDF' else user_icon_base64
        div = f"""
        <div style="display:flex; flex-direction: row; margin:5px"
    {'' if chat.origin == 'PDF' else 'row-reverse'}">
    <img style = "margin:3px 5px;" class="chat-icon" src="data:image/png;base64,{icon_base64}" width=32 height=32>
    <div class="chat-bubble
    {'ai-bubble' if chat.origin == 'PDF' else 'human-bubble'}" style="font-family: Arial, sans-serif; font-size: 18px; color: white;">
        &#8203;{chat.message} 
    </div>
</div>
        """
        st.markdown(div, unsafe_allow_html=True)
    
    for _ in range(3):
        st.markdown("")

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    query = cols[0].text_input(
        "Chat",
        
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit", 
        type="primary", 
        on_click=on_click_callback, 
    )

    


