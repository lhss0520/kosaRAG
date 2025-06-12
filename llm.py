from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from config import answer_examples


index_name = 'tax-index'

store = {}

#세션 ID를 기반으로 채팅 기록을 저장하고 불러오는 역할
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store: #세션 ID가 저장소에 없으면(채팅기록이없으면) 새로 생성
        store[session_id] = ChatMessageHistory()
    return store[session_id]


#질문을 사전에 따라 변환해주는 LLM 파이프라인을 구성하는 함수
#질문 → 사전에 따라 변경 (필요 시) → LLM 처리 → 최종 문자열로 출력
def get_dictionary_chain():
    dictionary = ['사람을 나타내는 표현 -> 거주자']
    llm = get_llm() #llm가져오는 함수 호출

    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해 주세요.
        만약 변경할 필요가 없다고 판단 된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해 주세요.
        사전: {dictionary}
        질문: {{question}}
        """)

    dictionary_chain= prompt | llm | StrOutputParser() 
    
    return dictionary_chain 


# llm을 가져오는 함수(질문을 보내고 답변을 받는 역할)
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    
    return llm


# Pinecone 벡터 스토어에서 검색기를 가져오는 함수
def get_retriever():
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = PineconeVectorStore.from_existing_index( 
        index_name=index_name,
        embedding=embedding,
    ) 
    
    retriever = vectorstore.as_retriever(search_kwargs={"k":4})
    
    return retriever


# 채팅 히스토리를 기반으로 질문을 재구성하는 검색기를 생성하는 함수
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), # 채팅 히스토리정보 들어가있음
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    return history_aware_retriever



def get_rag_chain():
    llm = get_llm()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples, #answer_examples : config.py에 정의된 예시들
    )    
    
    history_aware_retriever = get_history_retriever()
    
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해 주세요."
        "아래에 제공된 문서를 활용해서 답변해 주시고"
        "답변을 알 수 없다면 모른다고 답변해 주세요."
        "답변을 제공할 때 소득세법 (xx조)에 따르면 이라고 시작하면서 답변해 주시고"
        "2~3문장 정도의 짧은 내용의 답변을 원합니다."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer') #answer키만 볼수있게설정

    return conversational_rag_chain
    

def get_ai_message(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain() # get_rag_chain()은 conversational_rag_chain 리턴함 
    
    tax_chain = {"input" : dictionary_chain} | rag_chain  #conversational_rag_chain보면 query가아니라 input임
    ai_message = tax_chain.stream(
        {
        "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )
    
    return ai_message