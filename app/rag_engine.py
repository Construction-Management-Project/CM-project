from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from app.config import OPENAI_API_KEY, CHROMA_DIR, LLM_MODEL, LLM_TEMPERATURE
from openai import OpenAIError

# Initialization
try:
    print("Loading OpenAIEmbeddings...")
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    print("Loading FAISS DB...")
    vectordb = FAISS.load_local(
        CHROMA_DIR,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

    print("Initializing ChatOpenAI...")
    llm = ChatOpenAI(
        temperature=LLM_TEMPERATURE,
        model_name=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        request_timeout=15
    )
except Exception as e:
    print("Initialization failed:", e)
    raise e

# RAG query function
def query_rag(question: str, history: list[str], k: int = 3) -> str:
    print("Input question:", question)

    try:
        print("Starting similarity search in FAISS DB")
        docs = vectordb.similarity_search(question, k=k)
        print(f"Retrieved {len(docs)} relevant documents")
    except Exception as e:
        print("similarity_search failed:", e)
        return "Error: Failed to search from FAISS DB."

    context = "\n\n".join([doc.page_content for doc in docs]) or "No relevant documents were found."

    messages = [
        SystemMessage(content="You are a construction safety manual expert chatbot. Always respond in English, even if the question or documents are in Korean. Your answers must be accurate and concise."),
        HumanMessage(content=f"Refer to the following content to answer the question:\n\n{context}")
    ]

    for idx, msg in enumerate(history):
        if idx % 2 == 0:
            messages.append(HumanMessage(content=msg))
        else:
            messages.append(AIMessage(content=msg))

    messages.append(HumanMessage(content=question))

    try:
        print("Requesting GPT response...")
        response = llm.invoke(messages)
        print("GPT response received")
        return response.content
    except OpenAIError as e:
        print("GPT API error:", e)
        return "Error: GPT failed to respond. Please try again later."
