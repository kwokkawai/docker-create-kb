import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="data/question_response.csv")
documents = loader.load()

# print(documents[0])
# print(len(documents))

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

#customer_message = """
#The solution should have a network/ portal for suppliers to receive POs and submit invoices.
#"""
#
#result = retrieve_info(customer_message)
#print(result)


# 3. Setup LLMChain & prompts

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
We are a world class customer support representative. 

Below is a question I received from customer:
{message}

Here is a list of best practies of how we respond to customer in similar scenarios:
{best_practice}
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation

def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

#message = """
#can i use the pet stroller for human
#"""

#response = generate_response(message)
#print(response)

# 5. Build an app with streamlit


def main():
    st.set_page_config(
        page_title="Customer response generator", page_icon=":bird:")

    st.header("Customer response generator :bird:")
    message = st.text_area("customer message")

    if message:
        st.write("Generating best practice message...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()