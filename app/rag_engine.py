from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from app.config import OPENAI_API_KEY, CHROMA_DIR, LLM_MODEL, LLM_TEMPERATURE

# 벡터 DB 로드
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

# LLM 로드
llm = ChatOpenAI(
    temperature=LLM_TEMPERATURE,
    model_name=LLM_MODEL,
    openai_api_key=OPENAI_API_KEY
)

def query_rag(question: str, history: list[str], k: int = 3) -> str:
    docs = vectordb.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

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

    response = llm(messages)
    return response.content
