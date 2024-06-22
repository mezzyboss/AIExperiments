from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
from dotenv import find_dotenv, load_dotenv
import textwrap

load_dotenv(find_dotenv())
#os.environ["OPENAI_API_KEY"] = "sk-proj-kDDvIR7jS7hJrFu4gJN0T3BlbkFJVbYhLKx8bfXmBZNKFYKE"
embeddings = OpenAIEmbeddings()


def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = chat_prompt | llm

    response = chain.invoke({
        "question": query, 
        "docs" : docs_page_content})
    #response = response.replace("\n", "")
    return response, docs


# Example usage:
video_url = "https://www.youtube.com/live/T_CO5rqBk9g?si=xRFoow8LXMh8zH_4"
db = create_db_from_youtube_video_url(video_url)

query = "According to the pastor what do people really need?"
response, docs = get_response_from_query(db, query)
text = response.pretty_repr()
print(textwrap.fill(text, width=80))
