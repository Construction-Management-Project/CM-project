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
        SystemMessage(content="너는 건설 안전 매뉴얼 전문가 챗봇이야. 질문에 정확하고 간결하게 한국어로 답변해."),
        HumanMessage(content=f"아래 내용을 참고해서 질문에 답해줘:\n\n{context}")
    ]

    for idx, msg in enumerate(history):
        if idx % 2 == 0:
            messages.append(HumanMessage(content=msg))
        else:
            messages.append(AIMessage(content=msg))

    messages.append(HumanMessage(content=question))

    response = llm(messages)
    return response.content
