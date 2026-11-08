# Streamlit UI Front-End
import os
import base64
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun


# 1. Page Configuration & Background Setup

st.set_page_config(page_title="IITK PG Rules Assistant", page_icon="🎓", layout="centered")
load_dotenv()

# Document Hyperlinks
FAQ_URL = "https://drive.google.com/file/d/1Sgsh7a_7_pKyr6YzKy2LiSiz-p96QEby/view?usp=sharing"
MANUAL_URL = "https://drive.google.com/file/d/1lmb6X882qgzsEzmZwCkrQMejKIiW4ynt/view?usp=sharing"

def get_secret_key(key_name):
    """Retrieve API keys seamlessly from either local .env or Streamlit Cloud Secrets."""
    val = os.getenv(key_name)
    if not val and hasattr(st, "secrets") and key_name in st.secrets:
        val = st.secrets[key_name]
    return val

def set_background(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        page_bg_img = f'''
        <style>
        .stApp {{
          background-image: url("data:image/jpg;base64,{bin_str}");
          background-size: cover;
          background-position: center;
          background-attachment: fixed;
        }}
        h1, h2, h3, h4, h5, h6, p, li, span, label, .stMarkdown {{
            color: #ffffff !important;
            text-shadow: 
                2px 2px 4px rgba(0, 0, 0, 0.9), 
                -1px -1px 4px rgba(0, 0, 0, 0.9),
                0px 0px 10px rgba(0, 0, 0, 1.0) !important;
        }}
        [data-testid="stBottom"] > div {{
            background-color: transparent !important;
        }}
        [data-testid="stChatInput"] {{
            background-color: rgba(0, 0, 0, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
        }}
        [data-testid="stChatMessage"] {{
            background-color: rgba(0, 0, 0, 0.65) !important;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(4px);
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("9227948.jpg")

# 2. Pipeline Initialization

@st.cache_resource
def initialize_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    if os.path.exists("./chroma_db/faq") and os.path.exists("./chroma_db/manual"):
        faq_vs = Chroma(persist_directory="./chroma_db/faq", embedding_function=embeddings, collection_name="faq_collection")
        manual_vs = Chroma(persist_directory="./chroma_db/manual", embedding_function=embeddings, collection_name="manual_collection")
    else:
        faq_docs = PyPDFLoader("./PGRulesFAQ.pdf").load()
        manual_docs = PyPDFLoader("./PG-Manual.pdf").load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        faq_splits = text_splitter.split_documents(faq_docs)
        manual_splits = text_splitter.split_documents(manual_docs)

        faq_vs = Chroma.from_documents(documents=faq_splits, embedding=embeddings, persist_directory="./chroma_db/faq", collection_name="faq_collection")
        manual_vs = Chroma.from_documents(documents=manual_splits, embedding=embeddings, persist_directory="./chroma_db/manual", collection_name="manual_collection")

    faq_retriever = faq_vs.as_retriever(search_kwargs={"k": 3})
    manual_retriever = manual_vs.as_retriever(search_kwargs={"k": 5})

    system_prompt = (
        "You are a highly analytical and precise academic research assistant. "
        "Use ONLY the following retrieved context to answer the question. "
        "Provide a comprehensive, well-structured answer that includes any relevant conditions, "
        "exceptions, or limits mentioned in the context. "
        "If the answer is not explicitly contained in the context, you must reply with exactly: "
        "'I do not know.' Do not guess, infer, or hallucinate.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

    chain_pool = []
    gemini_keys = [get_secret_key("GEMINI_API_KEY_1"), get_secret_key("GEMINI_API_KEY_2")]
    valid_g_keys = [k for k in gemini_keys if k]
    gemini_models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-1.5-flash"]
    
    for key in valid_g_keys:
        for m in gemini_models:
            llm = ChatGoogleGenerativeAI(model=m, google_api_key=key, temperature=0, max_retries=0)
            chain_pool.append({
                "name": f"Google: {m}",
                "llm": llm,
                "chain": create_stuff_documents_chain(llm, prompt)
            })
            
    groq_key = get_secret_key("GROQ_API_KEY")
    if groq_key:
        groq_llm = ChatGroq(model="llama3-8b-8192", api_key=groq_key, temperature=0, max_retries=0)
        chain_pool.append({
            "name": "Groq: Llama 3 (8B)",
            "llm": groq_llm,
            "chain": create_stuff_documents_chain(groq_llm, prompt)
        })

    web_search = DuckDuckGoSearchRun()
    return faq_retriever, manual_retriever, chain_pool, web_search

faq_retriever, manual_retriever, chain_pool, web_search_tool = initialize_rag_pipeline()

# 3. Hierarchical Search Logic

def expand_query(query):
    expansion_prompt = (
        "You are a strict query normalizer for IIT Kanpur academic policies. "
        "Standardize degree names and slang terms in the user query to match official IITK PG Manual terms. "
        "DO NOT add extra concepts. ONLY return the concise normalized query.\n\n"
        "Rules:\n"
        "- Standardize degrees: MTech/MT -> M.Tech. | PhD/phd -> Ph.D. | MSR/MS -> M.S. by Research\n"
        "- Keep 'MSPD' exactly as 'MSPD'.\n"
        "- Map slang: CG/CGPA/marks -> CPI | stipend/salary -> financial assistantship | guide -> thesis supervisor.\n\n"
        f"Original Query: {query}\n"
        "Normalized Query:"
    )
    error_logs = []
    for pool_item in chain_pool:
        try:
            return pool_item["llm"].invoke(expansion_prompt).content.strip()
        except Exception as e:
            error_logs.append(f"{pool_item['name']} failed: {str(e)}")
            continue
            
    raise Exception(f"RESOURCE_EXHAUSTED. Hidden Errors: {error_logs}")

def hierarchical_rag_search(query, use_web_search):
# Guardrail: Handle Bot Self-Introduction Queries
    # Strip out punctuation so "Hi!" or "Who are you?" match perfectly
    clean_q = query.lower().strip().replace("?", "").replace("!", "").replace(".", "")
    
    intro_triggers = [
        "who are you", "what are you", "what can you do", 
        "what is this bot", "help", "help me", 
        "hi", "hello", "hey"
    ]
    
    # EXACT MATCH: Only trigger if the user's entire query is in the trigger list
    if clean_q in intro_triggers:
        return {
            "answer": (
                "👋 **Hello! I am the IITK PG Rules & FAQ Assistant.**\n\n"
                "I am an AI assistant designed to help postgraduate students navigate academic policies at **IIT Kanpur**.\n\n"
                "**What I can help you with:**\n"
                "- 📜 **PG Manual Regulations:** Minimum CPI rules, assistantship/stipend eligibility, thesis submission timelines, course waiver conditions, and degree extensions.\n"
                "- ❓ **FAQ Queries:** Common postgraduate administrative procedures and guidelines.\n"
                "- 🌐 **Web Search Fallback:** External official information regarding IIT Kanpur if rules aren't found in internal documents."
            ),
            "source_type": "System Bot Intro",
            "model_used": "Built-in Guardrail",
            "context_used": []
        }

    search_query = expand_query(query)
    
    def run_qa_chain(context):
        for pool_item in chain_pool:
            try:
                response = pool_item["chain"].invoke({"input": query, "context": context})
                return response, pool_item["name"]
            except:
                continue
        raise Exception("RESOURCE_EXHAUSTED")

    def run_web_llm(web_prompt):
        for pool_item in chain_pool:
            try:
                response = pool_item["llm"].invoke(web_prompt).content
                return response, pool_item["name"]
            except:
                continue
        raise Exception("RESOURCE_EXHAUSTED")

    # Step A: FAQ
    faq_context = faq_retriever.invoke(search_query)
    faq_response, model_used = run_qa_chain(faq_context)
    
    if "i do not know" in faq_response.lower().strip().replace(".", ""):
        # Step B: PG Manual
        manual_context = manual_retriever.invoke(search_query)
        manual_response, model_used = run_qa_chain(manual_context)
        
        if "i do not know" in manual_response.lower().strip().replace(".", ""):
            # Step C: Web Search
            if use_web_search:
                iitk_web_query = f"{query} IIT Kanpur" if "iit" not in query.lower() else query
                web_results = web_search_tool.invoke(iitk_web_query)
                web_prompt = f"Use these web results regarding IIT Kanpur to answer:\n\nQuery: {query}\n\nResults: {web_results}"
                final_web_answer, model_used = run_web_llm(web_prompt)
                
                return {
                    "answer": f"**As your query was not found in the IITK FAQ or Manual, I searched the web:**\n\n{final_web_answer}",
                    "source_type": "Web Search",
                    "model_used": model_used,
                    "context_used": []
                }
            else:
                return {
                    "answer": "I do not know. The answer is not in the provided documents, and web search is disabled.",
                    "source_type": "None",
                    "model_used": model_used,
                    "context_used": []
                }
                
        return {"answer": manual_response, "source_type": "PG Manual", "model_used": model_used, "context_used": manual_context}
    
    return {"answer": faq_response, "source_type": "FAQ", "model_used": model_used, "context_used": faq_context}

# 4. Streamlit UI Layout

st.title("🎓 IITK PG Rules & FAQ Assistant")
st.caption("Hierarchical RAG System with FAQ, PG Manual, and Enriched Web Fallback")

use_web = st.toggle("🌐 Enable Web Search Fallback", value=False)
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if user_prompt := st.chat_input("Ask a question about IITK PG rules..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching IITK policy documents..."):
            try:
                result = hierarchical_rag_search(user_prompt, use_web_search=use_web)
                
                formatted_response = f"{result['answer']}\n\n---\n"
                
                # Add Hyperlinked Citations
                if result['source_type'] == "FAQ":
                    formatted_response += f"**Source Document:** [PGRules FAQ]({FAQ_URL})\n\n"
                    for i, doc in enumerate(result['context_used']):
                        page_num = doc.metadata.get('page', 0) + 1
                        formatted_response += f"- Page {page_num}\n"
                elif result['source_type'] == "PG Manual":
                    formatted_response += f"**Source Document:** [PG Manual]({MANUAL_URL})\n\n"
                    for i, doc in enumerate(result['context_used']):
                        page_num = doc.metadata.get('page', 0) + 1
                        formatted_response += f"- Page {page_num}\n"
                else:
                    formatted_response += f"**Source:** {result['source_type']}\n"
                
                formatted_response += f"\n<br>\n\n**Generated by:** `{result['model_used']}`"
                
                st.markdown(formatted_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    st.error("🚨 **API Quota Exhausted!** \n\nAll primary and fallback LLMs have reached their daily rate limits. Please try again tomorrow.")
                else:
                    st.error(f"An unexpected error occurred: {str(e)}")
