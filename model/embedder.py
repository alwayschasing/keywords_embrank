import numpy as np
import gensim
class Word2VecEmbedder():
    """
        ELMo
        https://allennlp.org/elmo
    """
    def __init__(self, word2vec_file):
        pass

    def get_tokenized_words_embeddings(self, sents_tokened):
        """
        @see EmbeddingDistributor
        :param tokenized_sents: list of tokenized words string (sentences/phrases)
        :return: ndarray with shape (len(sents), dimension of embeddings)
        """
        max_len = max([len(sent) for sent in sents_tokened])
        elmo_embedding = self.elmo.sents2elmo(sents_tokened,output_layer=-2)
        elmo_embedding = [np.pad(emb, pad_width=((0,0),(0,max_len-emb.shape[1]),(0,0)) , mode='constant') for emb in elmo_embedding]
        elmo_embedding = torch.from_numpy(np.array(elmo_embedding))
        return elmo_embedding

# if __name__ == '__main__':
#     sents = [['今', '天', '天气', '真', '好', '啊'],
#              ['潮水', '退', '了', '就', '知道', '谁', '没', '穿', '裤子']]
#     elmo = WordEmbeddings()
#     embs = elmo.get_tokenized_words_embeddings(sents)
#     print("OK")