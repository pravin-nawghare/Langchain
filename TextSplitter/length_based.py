from langchain_classic.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0, 
# chunk_overlap --> in CharacterTextSplitter after chunk size the text gets splitted even if words are not completed. So the context gets missing and to have some similar context betweem two chunks chunk_overlap is used. For RAG based application the value lies between 10 - 20% of chunk_size 
    separator=''
)

docs = '''
after waking up take 50g of bamboo muraba (30min before breakfast). after 2hr of breakfast in camel milk powder(1.5 tablespoon) add the mixture 1 teaspoon (mixture- 400g ashwangandha, 100g shatavari, 500g thread mishri). water should be luke warm(200 ml). do not eat for 2 hrs after having it. fresh curd, fresh butter, panner, ripe bananas,  ripe mangoes, dates, chiku, black urad dal. tadasana for 5min, chakrasna for 5min, suryanamskar for 5min (either before step 1 or at evening). upright posture, massage, nofap, acupressure point(upper part of thumb), sunlight.
2hr after dinner (or 30min before sleep) have milk with the mixture and focus on third eye immediately before sleeping for 2min then to 5 min
'''

split = splitter.split_text(docs)
# split = splitter.split_documents(docs) --> for documents

# print(split)
i = 1
for split_text in split:
    i += 1
    print(str(i)+' '+split_text)