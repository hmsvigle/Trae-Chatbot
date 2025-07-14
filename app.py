import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Set page configuration
st.set_page_config(page_title="Simple Chatbot", page_icon="🤖")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = None

@st.cache_resource
def load_model():
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # Reduced for faster responses
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    
    # Create LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Initialize conversation chain with memory
    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory()
    )
    
    return conversation

# Main UI
st.title("🤖 Simple Chatbot")
st.write("Chat with TinyLlama - a lightweight chatbot!")

# Load the model
if st.session_state.conversation is None:
    with st.spinner("Loading the model... This might take a while..."): 
        st.session_state.conversation = load_model()

# Chat interface
user_input = st.text_input("You:", key="user_input")

if user_input:
    with st.spinner("Thinking..."):
        # Get response from the model
        response = st.session_state.conversation.predict(input=user_input)
        
        # Display the response
        st.write("Bot: ", response)

# Display conversation history
with st.expander("View Conversation History"):
    history = st.session_state.conversation.memory.buffer
    st.write(history)