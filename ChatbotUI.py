__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import openai
import json
from openai import OpenAI
from streamlit_feedback import streamlit_feedback

COHERE_KEY = st.secrets['COHERE_KEY']
openai_api_key = st.secrets['OPENAI_API_KEY']

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="feedback_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    
    
##### CONNECT TO DATABASE and OpenAI #####
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = 'chromadb_major_travel/'
COLLECTION_NAME = "document_embeddings"

def text_embedding(text):
    response = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return response["data"][0]["embedding"]

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )

client = chromadb.PersistentClient(path = CHROMA_DATA_PATH)
collection = client.get_or_create_collection(
    name = COLLECTION_NAME,
    embedding_function = openai_ef,
    metadata = {"hnsw:space" : "cosine"}
)

OpenAIClient = openai.OpenAI(
    api_key=openai_api_key,
)
###############################

#### DEFINE FUNCTION CALLS ##############
import cohere
co = cohere.Client(COHERE_KEY)
def get_relevant_context(query, limit = 5):
    relevant_context = collection.query(
        query_texts = [query],
        n_results = limit,
        include=["documents","distances","metadatas"]
    )
    
    distance_threshold = 0.6
    documents = []
    metadatas = []
    for dist_lst, document_lst, meta_lst in list(zip(relevant_context['distances'], relevant_context['documents'], relevant_context['metadatas'])):
        for dst, doc, meta in list(zip(dist_lst, document_lst, meta_lst)):
            if dst <= distance_threshold:
                documents.append(doc) 
                metadatas.append(meta)
        
                         
    if documents:
        index2doc = {doc : i for i,doc in enumerate(documents)}
        results = co.rerank(query=query, documents=documents, top_n=2, model='rerank-english-v3.0', return_documents=True)   
        documents = [str(r.document.text) for r in results.results] 
        document_indexes = [index2doc[doc] for doc in documents]
        filenames = [metadatas[i] for i in document_indexes]
        
        FN_DOC = [f"CONTEXT_SOURCE_FILE:{file}\nCONTENT:{docu}\n" for file,docu in list(zip(filenames, documents)) ]
        context_data = "\n".join(FN_DOC)
        context_str = f"You may use the following SOP Documents to answer the question:\n{context_data}"
        return context_str
    else:
        return "NO RELEVANT CONTEXT FOUND"

SIGNATURE_get_relevant_context = {
    "type" : "function",
    "function" : {
        "name" : "get_relevant_context",
        "description" : "Get related SOPs to use as context from ChromaDB",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "query" : {
                    "type" : "string",
                    "description" : "Query passed by the user to the chatbot"
                },
                "limit" : {
                    "type" : "integer",
                    "description" : "Total number of SOPs to retrieve from vector database"
                }
            },
            "required" : ["query"],
        }
    }
    
}

tools = [SIGNATURE_get_relevant_context]

#######################################


###### ADD CUSTOM FUNCTIONS ###########
import tiktoken
def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += 4
        try:
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += -1
        except:
            num_tokens += len(encoding.encode(str(message)))
    num_tokens += 2
    return num_tokens

############### PROMPTS ###############

system_prompt = """
As a travel agent assistant for Major Travel, your role involves strictly adhering to the agency's standard operating procedures (SOPs) and internal tasks to ensure high-quality service delivery. Your primary objective is to support senior colleagues in identifying the most relevant references based on company SOPs and assisting them with their daily tasks.

If the requested information is not found in the provided documents, you have three options:
1. If it's the initial query and you lack specific details to provide a precise answer, ask the user for additional information to better address their query.
2. For follow-up queries where the current chat context is insufficient, inform the user that the current context cannot adequately address their query and utilize the function `get_relevant_context` to search for more relevant SOPs.
3. If there is no relevant context found, simply say that the information cannot be found within the company's SOPs.

Answer ONLY with the facts extracted from the ChromaDB. If there isn't enough information, say you don't know. Do not generate answers that don't use the sources provided to you. If asking a clarifying question to the user would help, ask the question.
To help in monitoring performance, include the CONTEXT_SOURCE_FILE of the relevant context extracted in the form of a header.
"""
########################################
    
st.title("📝 Major Travel Chatbot UAT Platform")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {'role' : 'system' , 'content' : system_prompt},
        {"role": "assistant", "content": "How can I help you? Leave feedback to help me improve!"}
    ]
    
if "response" not in st.session_state:
    st.session_state["response"] = ''
    
messages = st.session_state.messages
for msg in messages:
    try:
        if msg['role'] not in ['system', 'tool']:
            if "QUERY_CLEAN" not in msg['content']:
                st.chat_message(msg['role']).write(msg['content'])
        else:
            #print(msg)
            pass
    except:
        pass
        #print(msg)
        
# delete older completions to keep conversation under token limit
while num_tokens_from_messages(messages) >= 8192*0.8:
    print("Removing Older Texts due to token number!")
    messages.pop(0)
print("Current number of Tokens : ",  num_tokens_from_messages(messages))
    
if prompt := st.chat_input(placeholder="What do you want to know about Major Travel's SOPs"):
    
    st.chat_message("user").write(prompt)

    
    messages.append({"role" : "user" , "content" : prompt})
       
    if not openai_api_key:
        st.info("Please add your OpenAI API Key to continue.")
        st.stop()
        
    # Get Response
    response = OpenAIClient.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=0,
        n=1,
        tools = tools,
        tool_choice = "auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        available_fxns = {
            "get_relevant_context" : get_relevant_context
        }
        
        messages.append(response_message),
        
        for tool_call in tool_calls:
            fxn_name = tool_call.function.name
            fxn_to_call = available_fxns[fxn_name]
            fxn_args = json.loads(tool_call.function.arguments)
            fxn_response = fxn_to_call(
                **fxn_args
            )
            
            messages.append(
                {
                    "tool_call_id" : tool_call.id,
                    "role" : "tool",
                    "name" : fxn_name,
                    "content" : fxn_response
                }
            )
    context_enhanced_response = OpenAIClient.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=0,
        n=1,
    )
    
    # Extract Answer
    answer = context_enhanced_response.choices[0].message.content
    st.session_state["response"] = answer
    with st.chat_message("assistant"):
        messages.append({"role" : "assistant", "content" : st.session_state["response"]})
        st.write(st.session_state["response"])
        
if st.session_state["response"]:
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        key = f"feedback_{len(messages)}"
    )
    if feedback:
        # Placeholder for logging
        
        print(feedback)
        st.toast("Feedback recorded!", icon="📝")
