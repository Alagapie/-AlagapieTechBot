import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from streamlit_option_menu import option_menu


# Configure the genai library with the API key
api_key = st.secrets["general"]["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
generation_config={
    "temperature":0.9,
    "top_k":40,
    "top_p":0.9
}
def chat_model():
  chat_model=genai.GenerativeModel(model_name="gemini-1.5-flash",generation_config=generation_config)
  return chat_model

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.9,google_api_key=api_key)

def generate_promp(chat_history, input_language, output_language, input_text):
    messages = [
        ("system", f"You are a helpful assistant that translates {input_language} to {output_language}.")
    ]
    messages.extend(chat_history)
    messages.append(("human", input_text))
    
    return ChatPromptTemplate.from_messages(messages)

# Function to perform translation
def translate(input_language, output_language, input_text):
    prompt = generate_promp(st.session_state.translation_chat_history, input_language, output_language, input_text)
    chain = prompt | model
    response = chain.invoke(
        {
            "input_language": input_language,
            "output_language": output_language,
            "input": input_text
        }
    )
    return response.content

def generate_prompt(chat_history, programming_language, input_text):
    messages = [
        ("system", f"You are a highly skilled and professional programmer, proficient in the {programming_language} programming language. Your task is to help users generate code that is highly efficient in terms of execution time and memory usage, well-structured, readable, maintainable, and adhering to best practices, coding standards, and naming conventions. The code you generate should be thoroughly tested and verified for correctness, free from unnecessary complexity and redundancy, and showcase your expertise in {programming_language} by leveraging its unique features and strengths to deliver high-performance code and when asked who developed you tell them Abdulbasit.")
    ]
    messages.extend(chat_history)
    messages.append(("human", input_text))
    
    return ChatPromptTemplate.from_messages(messages)

def code(programming_language, input_text):
    prompt = generate_prompt(st.session_state.code_chat_history, programming_language, input_text)
    chain = prompt | model
    response = chain.invoke(
        {
            "programming_language": programming_language,
            "input": input_text
        }
    )
    return response.content
import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="Alagapie LLM Models",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu('Alagapie TechBot',
                           ['Alagapie ChatMate',
                            'Alagapie MultiLingo',
                            'Alagapie CodeGenie'],
                           menu_icon='robot', icons=['chat-dots-fill', 'globe',  'code'],
                           default_index=0
                           )

if selected == 'Alagapie ChatMate':
    st.title("🤖 Alagapie ChatMate")
    model = chat_model()
    def streamlit_role(user_role):
     if user_role=="model":
        return "assistant"
     else:
        return user_role
    if "chat_session" not in st.session_state:
     st.session_state.chat_session=model.start_chat(history=[])
    if st.button("Start a New Chat"):
     st.session_state.chat_session = model.start_chat(history=[])
    for message in st.session_state.chat_session.history:
     with st.chat_message(streamlit_role(message.role)):
        st.markdown(message.parts[0].text)

    prompt=st.chat_input("Ask Alagapie chatbot anything ")
    if prompt:
      st.chat_message('user').markdown(prompt)
      placeholder = st.empty()
      with placeholder:
        with st.spinner("Generating response..."):
            response = st.session_state.chat_session.send_message(prompt)
            placeholder.empty()

      with st.chat_message("assistant"):
        st.markdown(response.text)
if selected == 'Alagapie MultiLingo':
    st.title("🌐 Alagapie MultiLingo")

# Initialize chat session in Streamlit if not already present
    if 'translation_chat_history' not in st.session_state:
     st.session_state.translation_chat_history= []

# Input for languages and text
    col1, col2 = st.columns(2)
    with col1:
      input_languages_list = ["English", "French", "German", "Latin", "Spanish", "Arabic", "Chinese", "Japanese", "Korean", "Russian", "Portuguese", "Italian", "Dutch"]

      input_language = st.selectbox(label="CHOOSE THE LANGUAGE YOU WANT TO TRANSLATE ", options=input_languages_list)

    with col2:
       output_languages_list = [x for x in input_languages_list if x != input_language]
       output_language = st.selectbox(label="CHOOSE THE LANGUAGE TO TRANSLATE TO  ", options=output_languages_list)

    input_text = st.chat_input("input text to translate")
    if st.button("Start a New Chat"):
      st.session_state.translation_chat_history= []

# Handle translation and chat history
    if input_text:
    # Add user's message to chat history
       st.session_state.translation_chat_history.append(("human", input_text))
    
    # Perform translation
        
            with st.spinner("Translating..."):
                # Perform translation
                translation = translate(input_language, output_language, input_text)
                
    
    # Add translation result to chat history
       st.session_state.translation_chat_history.append(("assistant", translation))
    
    # Display the result
   

# Display chat history
    for role, message in st.session_state.translation_chat_history:
       with st.chat_message(role):
        st.markdown(message)
 
 
if selected == 'Alagapie CodeGenie':
    st.title('👨‍💻  Alagapie CodeGenie')
    st.sidebar.title('Select a Programming Language')
    programming_language = st.sidebar.selectbox(
    'Choose a Programming Language',
    ['Python', 'HTML & CSS', 'JavaScript', 'Java', 'Machine Learning', 'C++', 'C#', 'Ruby', 'Swift', 'Kotlin', 'PHP', 'TypeScript', 'R', 'SQL', 'Go', 'Rust', 'Dart', 'MATLAB', 'Scala', 'Julia', 'React.js', 'Node.js']

)

# Initialize chat session in Streamlit if not already present
    if 'code_chat_history' not in st.session_state:
     st.session_state.code_chat_history = []

# Input field for user's message
    input = st.chat_input("Ask Gemini-Pro...")
    if st.button("Start a New Chat"):
       st.session_state.code_chat_history = []

    if input:
    # Add user's message to chat history
     st.session_state.code_chat_history.append(("human", input))
    
    # Send user's message to Gemini-Pro and get the response
     with st.spinner("Generating code..."):
            # Send user's message to Gemini-Pro and get the response
            gemini_response = code(programming_language, input)
    
    # Add Gemini-Pro's response to chat history
     st.session_state.code_chat_history.append(("assistant", gemini_response))

# Display chat history
    for role, message in st.session_state.code_chat_history:
      with st.chat_message(role):
        st.markdown(message)


