from langchain_community.document_loaders import WebBaseLoader
# for static web page WebBaseLoader
# for javascript heavy (user action changes page like flipkart)  use SeleniumURLLoader

url = 'prvide url'
loader = WebBaseLoader(url)
# with one url one web page will loaded
# but with multiple url provided in a list all web pages can be loaded

docs = loader.load()
print(docs[0].page_content)