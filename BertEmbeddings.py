from bert_embedding import BertEmbedding

class BertEmbeddings:

    def __init__(self,model = 'bert_24_1024_16',corpus='book_corpus_wiki_en_cased'):
        self.__model = model
        self.__corpus = corpus

        assert self.__model in ['bert_12_768_12','bert_24_1024_16'], "Model is not recognized."
        assert self.__corpus in ['book_corpus_wiki_en_uncased','book_corpus_wiki_en_cased','wiki_multilingual','wiki_multilingual_cased'], "Corpus is unknown."

        self.__bert = BertEmbedding(model = self.__model, dataset_name = self.__corpus)


    def predict(self,text):
        if not isinstance(text,list):
            text = [text]

        bertEmbeddings = self.__bert.embedding(text)
        return bertEmbeddings


import pandas as pd
texts = ['how do I learn <learnings>','what are the ways to get trained on <learnings>']
obj = BertEmbeddings()
A = obj.predict(texts)
B = pd.DataFrame(A,columns = ['A','B'])
print(B.shape)
A = obj.predict(texts)
B = pd.DataFrame(A,columns = ['A','B'])
print(B.shape)
