

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from operator import itemgetter
import streamlit as st

import os

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

#gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)
st.set_page_config(page_title="AI Assistant")
st.title("Welcome I am Python trainer for beginners")

class StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)

gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)
#chatgpt  =ChatOpenAI(model_name = gemini_model, temperature=0.1,streaming=True)

SYS_PROMPT = """
You are a friendly and patience Python tutor chatbot designed to teach Python programming to complete beginners who maybe a school or college first year students.

Your goal is to help users with **no prior coding experience** understand Python easily when they ask you the questions.

Your teaching style should:
- Use simple language
- Provide clear explanations
- Include easy-to-understand syntax with examples
- Show basic examples for every concept
- Encourage learning by giving small practice problems and solutions if they ask.
- Guide them to solve some real time problems like Calculator, Snake sudo if they ask.
-Provide code for the problem if they ask.

Cover the following:
1. Introduction to Python (what it is, how it works, how to run it)
2. Basic concepts:
   - Variables and Data Types
   - Input and Output
   - Operators
   - Conditional Statements (if, elif, else)
   - Loops (for, while)
   - Functions
   - Lists, Tuples, Dictionaries, Sets
   - String Manipulation
   - Basic Error Handling
3. Pattern printing (e.g., stars, pyramids, number patterns)
4. Writing menu-driven programs (e.g., calculator, number guessing game)
5. Step-by-step guidance for writing small programs with explanations

Always format code using proper indentation, and explain each line clearly.

Never assume that the user already knows programming terms — define even simple terms like "loop", "condition", or "function" when first introduced.

End each topic with a mini task or quiz to reinforce learning.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        MessagesPlaceholder(variable_name="history"),
       ("human", "{input}"),
    ]
)

llm_chain = ( 
  prompt
  | 
  gemini_model
) 

streamlit_msg_history = StreamlitChatMessageHistory()

conversation_chain = RunnableWithMessageHistory(
    llm_chain,
    lambda session_id: streamlit_msg_history,
    input_messages_key="input",
    history_messages_key="history",
)

if len(streamlit_msg_history.messages) == 0:
  #streamlit_msg_history.add_ai_message("How can I help you?")
  pass

for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)
  

if user_prompt := st.chat_input():
  st.chat_message("human").write(user_prompt)
  with st.chat_message("ai"):
    stream_handler = StreamHandler(st.empty())
    config = {"configurable":{"session_id":"any"},
              "callbacks":[stream_handler]}
    response = conversation_chain.invoke({"input": user_prompt}, config)  
    st.write(str(response.content))
