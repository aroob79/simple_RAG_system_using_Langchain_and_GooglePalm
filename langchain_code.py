
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from keys import api_keys


def llm_loading(api_keys, urls, text_box):
    llm = GoogleGenerativeAI(model="models/text-bison-001",
                             google_api_key=api_keys, temperature=0.5)
    # initiate the splitter
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", "\n"],
        chunk_size=1000,
        chunk_overlap=200
    )

    # initiate the embedding
    google_embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_keys)

    # first load the content from the website
    text_box.text('loading the data........')
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_box.text('Embedding the data........')
    splitted_docs = splitter.split_documents(data)

    # initate the vector database
    vectdb = FAISS.from_documents(splitted_docs, google_embedding)
    retriever = vectdb.as_retriever()
    chai_ = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        input_key='query',
        return_source_documents=True)
    return chai_


def llm_QA(question, chai_):

    results = chai_({'query': question})
    return results['result'], [results['source_documents'][i].metadata['source'] for i in range(len(results['source_documents']))]


if __name__ == "__main__":
    qapalm = QAGooglePalm(api_keys)
    urls = [
        "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)",
        "https://cleartax.in/s/world-gdp-ranking-list",
        "https://www.focus-economics.com/countries/bangladesh/"]
    chain = qapalm.llm_loading(urls)
    ques = "Give a overal summary of the condition of bangladesh economy"
    print(qapalm.llm_QA(ques, chain))
