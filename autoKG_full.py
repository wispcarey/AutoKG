import sys

import numpy as np
import openai
from annoy import AnnoyIndex
from scipy import sparse
from scipy.sparse import diags, csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import graphlearning as gl
import tiktoken
import string
import time

# graph visualization
import networkx as nx
import matplotlib.pyplot as plt

from langchain.embeddings.openai import OpenAIEmbeddings

from utils import *

class autoKG():
    def __init__(self, texts: list, source: list, embedding_model: str, llm_model: str, openai_api_key: str,
                 main_topic: str,
                 embedding: bool = True):
        openai.api_key = openai_api_key
        self.texts = texts
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.source = source

        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        if embedding:
            self.vectors = np.array(self.embeddings.embed_documents(self.texts))
        else:
            self.vectors = None

        self.weightmatrix = None
        self.graph = None
        self.encoding = tiktoken.encoding_for_model(llm_model)
        if texts is None:
            self.token_counts = None
        else:
            self.token_counts = get_num_tokens(texts, model=llm_model)

        self.separator = "\n* "
        self.main_topic = main_topic

        # keywords graph
        self.keywords = None
        self.keyvectors = None
        self.U_mat = None
        self.pred_mat = None
        self.A = None
        self.dist_mat = None

        # completion parameters
        self.temperature = 0.1
        self.top_p = 0.5

    def get_embedding(self, text):
        result = openai.Embedding.create(
            model=self.embedding_model,
            input=text
        )
        return result["data"][0]["embedding"]

    def update_keywords(self, keyword_list):
        self.keywords = keyword_list
        self.keyvectors = np.array(self.embeddings.embed_documents(self.keywords))

    def make_graph(self, k, method='annoy', similarity='angular', kernel='gaussian'):
        knn_data = gl.weightmatrix.knnsearch(self.vectors, k, method, similarity)
        W = gl.weightmatrix.knn(None, k, kernel, symmetrize=True, knn_data=knn_data)
        self.weightmatrix = W
        self.graph = gl.graph(W)

    def remove_same_text(self, use_nn=True, n_neighbors=5, thresh=1e-6, update=True):
        to_delete = set()
        to_keep_set = set()

        if use_nn:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine').fit(self.vectors)
            distances, indices = nbrs.kneighbors(self.vectors)

            for i in range(self.vectors.shape[0]):
                for j, distance in zip(indices[i], distances[i]):
                    if i != j and distance < thresh and i not in to_delete and j not in to_delete:
                        if self.token_counts[i] >= self.token_counts[j]:
                            to_delete.add(i)
                            if i in to_keep_set:
                                to_keep_set.remove(i)
                            if j not in to_delete:
                                to_keep_set.add(j)
                        else:
                            to_delete.add(j)
                            if j in to_keep_set:
                                to_keep_set.remove(j)
                            if i not in to_delete:
                                to_keep_set.add(i)
        else:
            D = pairwise_distances(self.vectors, metric='cosine')

            for i in range(self.vectors.shape[0]):
                for j in range(i + 1, self.vectors.shape[0]):
                    if D[i, j] < thresh:
                        if self.token_counts[i] >= self.token_counts[j]:
                            to_delete.add(i)
                            if i in to_keep_set:
                                to_keep_set.remove(i)
                            if j not in to_delete:
                                to_keep_set.add(j)
                        else:
                            to_delete.add(j)
                            if j in to_keep_set:
                                to_keep_set.remove(j)
                            if i not in to_delete:
                                to_keep_set.add(i)

        all_indices = set(range(self.vectors.shape[0]))
        to_keep = np.array(list(all_indices - to_delete)).astype(int)
        to_delete = np.array(list(to_delete)).astype(int)
        remains = np.array(list(to_keep_set))

        if update:
            self.texts = [self.texts[i] for i in to_keep]
            self.source = [self.source[i] for i in to_keep]
            self.vectors = self.vectors[to_keep]
            self.token_counts = [self.token_counts[i] for i in to_keep]

        return to_keep, to_delete, remains

    def core_text_filter(self, core_list, max_length):
        if self.llm_model.startswith(("gpt-3.5")):
            model = "gpt-3.5-turbo-16k"
        else:
            model = "gpt-4"

        header = f"""
You are an advanced AI assistant, specialized in analyzing various pieces of information \
and providing precise summaries. In the following task, you will be provided with a list of keywords, \
each separated by a comma (,). You can split or reorganize the terms according to your understanding.

You should obey the following rules when doing this task:
1, Keywords in your answer should related to the topic {self.main_topic};
2, Remove duplicate or semantically similar terms;
3, Each keyword should be at most {max_length} words long;
4, Don't include any other symbols in each of your keyword;
"""

        examples = f"""
Here are two examples as your reference:  
  
Raw Keywords: 
Mineral processing EPC,Mineral processing,EPC Mineral processing..,circulation,suction,hollow shaft.,alkali-acid purification

Processed Keywords: 
Mineral processing EPC,circulation,suction,hollow shaft,alkali-acid purification

Raw Keywords: 
1. Mineral processing EPC 
2. Stoping methods 
3. Open pit mining 
4. Stripping ratio 
5. Small-scale mining 

Processed Keywords: 
Mineral processing EPC,Stoping methods,Open pit mining,Stripping ratio,Small-scale mining
"""
        prompt = f"""
{header}        

{examples}

Your task is to process following raw keywords:
Raw Keywords:      
{",".join(core_list)}  

Processed Keywords:
"""

        input_tokens = len(self.encoding.encode(",".join(core_list)))
        response, _, tokens = get_completion(prompt,
                                             model_name=model,
                                             max_tokens=input_tokens,
                                             temperature=self.temperature,
                                             top_p=self.top_p)
        response = response[:-1] if response.endswith(".") else response

        process_keywords = response.strip().split(',')

        return process_keywords, tokens

    def sub_entry_filter(self):
        # remove some keywords such that there is not any pair of keywords that one is the substring of the other
        if self.keywords is None:
            raise ValueError("Please extract keywords first.")
        strings = self.keywords.copy()

        i = 0
        while i < len(strings):
            for j in range(len(strings)):
                if i != j and strings[i] in strings[j]:
                    strings.pop(j)
                    if j < i:
                        i -= 1
                    break
            else:
                i += 1

        i = len(strings) - 1
        while i >= 0:
            for j in range(len(strings) - 1, -1, -1):
                if i != j and strings[i] in strings[j]:
                    strings.pop(j)
                    if j < i:
                        i -= 1
                    break
            else:
                i -= 1

        self.keywords = strings
        self.keyvectors = np.array(self.embeddings.embed_documents(self.keywords))
        return strings

    def final_keywords_filter(self):
        if self.keywords is None:
            raise ValueError("Please extract keywords first.")

        header = f"""
You have been provided a list of keywords, each separated by a comma. Your task is to process this list according to a set of guidelines designed to refine and improve the list's utility. 

Upon completion of your task, the output should be a processed list of keywords. Each keyword should be separated by a comma. 

It's crucial that you maintain the same formatting (separate keywords by comma) to ensure the usability of the processed list.

"""
        task1 = f"""'Concentration and Deduplication':
Your task is to examine each keyword in the list and identify those that are either identical or extremely similar in meaning. These should be treated in two ways:

- For keywords that are closely related, expressing different aspects of the same core idea, you need to 'concentrate' them into a single term that best captures the overall concept. For instance, consider the keywords <Styrofoam recycling>, <Styrofoam packaging recycling>, <Styrofoam recycling machine>, <Foam recycling>, <recycled Styrofoam products>. These all relate to the central concept of 'Styrofoam recycling' and should be consolidated into this single keyword.

- For keywords that are identical or nearly identical in meaning, remove the duplicates so that only one instance remains. The guideline here is: if two keywords convey almost the same information, retain only one. 

Remember, the objective of this step is to trim the list by eliminating redundancy and focusing on the core concepts each keyword represents. 
"""

        task2 = f"""'Splitting':
Sometimes, a keyword might be made up of an entity and another keyword, each of which independently conveys a meaningful concept. In these cases, you should split them into two separate keywords. 

For instance, consider a keyword like 'Apple recycling'. Here, 'Apple' is a distinct entity, and 'recycling' is a separate concept. Therefore, it's appropriate to split this keyword into two: 'Apple' and 'recycling'.

However, when you split a keyword, be sure to check if the resulting terms are already present in the list you have generated. If they are, remove the duplicates to ensure the list remains concise and unique. Always aim to avoid redundancy in the list. 
"""
        task3 = f"""'Deletion':

You will encounter keywords that are either too vague or represent an overly broad concept. These should be removed from the list.

For instance, a keyword like 'things' or 'stuff' is generally too vague to be useful. Similarly, a keyword like 'technology' might be too broad unless it's used in a specific context or in conjunction with other terms.
"""
        reminder = f"""
As you generate the processed list of keywords, constantly compare them with the original list and the keywords you've already processed, to ensure the integrity and consistency of your work.

Don't response anything other than processed keywords. Each of your processed keyword should have at most 3 words.
"""

        keyword_string = ",".join(self.keywords)

        if self.llm_model.startswith("gpt-3.5"):
            model = "gpt-3.5-turbo-16k"
        else:
            model = self.llm_model

        def quick_prompt(keyword_string, task):
            return f"""
{header}

Input Keywords:
{keyword_string}   

Instructions:
{task}

{reminder}

Your processed keywords:     
"""

        all_tokens = 0

        num_tokens = len(self.encoding.encode(keyword_string))
        keyword_string, _, tokens = get_completion(quick_prompt(keyword_string, task1),
                                                   model_name=model,
                                                   max_tokens=num_tokens,
                                                   temperature=self.temperature,
                                                   top_p=self.top_p)
        all_tokens += tokens

        num_tokens = len(self.encoding.encode(keyword_string))
        keyword_string, _, tokens = get_completion(quick_prompt(keyword_string, task2),
                                                   model_name=model,
                                                   max_tokens=num_tokens,
                                                   temperature=self.temperature,
                                                   top_p=self.top_p)
        all_tokens += tokens

        num_tokens = len(self.encoding.encode(keyword_string))
        keyword_string, _, tokens = get_completion(quick_prompt(keyword_string, task3),
                                                   model_name=model,
                                                   max_tokens=num_tokens,
                                                   temperature=self.temperature,
                                                   top_p=self.top_p)
        all_tokens += tokens

        self.keywords = keyword_string.split(",")
        self.keyvectors = np.array(self.embeddings.embed_documents(self.keywords))
        return keyword_string, all_tokens

    def summary_contents(self, indx, sort_inds, avoid_content=None,
                         max_texts=5, prompt_language='English',
                         num_topics=3, max_length=3, show_prompt=False):

        if avoid_content is None:
            avoid_content = []

        if self.llm_model.startswith("gpt-3.5"):
            model = "gpt-3.5-turbo-16k"
        else:
            model = self.llm_model

        if model.startswith("gpt-3.5"):
            max_num_tokens = 11900
        elif model.startswith("gpt-4"):
            max_num_tokens = 7900
        else:
            raise ValueError("Model should be either GPT-3.5 or GPT-4.")

        header = f"""
You are an advanced AI assistant, specialized in analyzing \
various pieces of information and providing precise summaries. Your task \
is to determine the core theme in the following series of *-separated \
information fragments, which is delimited by triple backticks. \
Ensure your answer focuses on the topic and avoids \
including unrelated content. DO NOT write complete sentences. 

You should obey the following rules when doing this task:
1, Keywords in your answer should related to the topic {self.main_topic};
2, Your answer should include at most {num_topics} keywords;
3, Each keyword should be at most {max_length} words long;
4, avoid already appeared theme keywords, marked inside <>;
5, Write your answer in {prompt_language};
6, Separate your output keywords with commas (,);
7, Don't include any other symbols other than keywords.
"""

        avoid_info = f"""
Please avoid the following already appeared theme terms: 
<{",".join(avoid_content)}>        
"""


        chosen_texts = []
        chosen_texts_len = 0
        chosen_texts_len += len(self.encoding.encode(header))
        chosen_texts_len += len(self.encoding.encode(avoid_info))

        separator_len = len(self.encoding.encode(self.separator))

        for i in range(min(max_texts, len(sort_inds))):
            select_text = [self.texts[j] for j in indx][sort_inds[i]]

            num_tokens = len(self.encoding.encode(select_text))
            chosen_texts_len += num_tokens + separator_len
            if chosen_texts_len > max_num_tokens - 500:
                break

            chosen_texts.append(self.separator + select_text)


        prompt = f"""
{header}

Information: 
'''{''.join(chosen_texts)}'''

{avoid_info}

Your response: 

"""
        if show_prompt:
            print(prompt)

        response, _, tokens = get_completion(prompt,
                                             model_name=model,
                                             max_tokens=200,
                                             temperature=self.temperature,
                                             top_p=self.top_p)
        return response, tokens

    ### cluster function
    def cluster(self, n_clusters, clustering_method='k_means', max_texts=5,
                select_mtd='similarity', prompt_language='English', num_topics=3,
                max_length=3, post_process=True, add_keywords=True, verbose=False):

        if clustering_method == 'k_means':
            kmeans_model = KMeans(n_clusters, init='k-means++', n_init=5)
            kmeans = kmeans_model.fit(np.array(self.vectors))
        elif clustering_method in ['combinatorial', 'ShiMalik', 'NgJordanWeiss']:
            extra_dim = 5
            if self.weightmatrix is None:
                self.make_graph(k=20)
            n = self.graph.num_nodes
            if clustering_method == 'combinatorial':
                vals, vec = self.graph.eigen_decomp(k=n_clusters + extra_dim)
            elif clustering_method == 'ShiMalik':
                vals, vec = self.graph.eigen_decomp(normalization='randomwalk', k=n_clusters + extra_dim)
            elif clustering_method == 'NgJordanWeiss':
                vals, vec = self.graph.eigen_decomp(normalization='normalized', k=n_clusters + extra_dim)
                norms = np.sum(vec * vec, axis=1)
                T = sparse.spdiags(norms ** (-1 / 2), 0, n, n)
                vec = T @ vec  # Normalize rows
            kmeans = KMeans(n_clusters, init='k-means++', n_init=5).fit(vec)
        else:
            raise ValueError(
                "Invalid clustering method. Choose 'k_means', 'combinatorial', 'ShiMalik' or 'NgJordanWeiss'.")

        all_tokens = 0
        cluster_names = []
        for i in range(len(kmeans.cluster_centers_)):
            center = kmeans.cluster_centers_[i]
            indx = np.arange(len(self.texts))[kmeans.labels_ == i]
            if select_mtd == 'similarity':
                if clustering_method == 'k_means':
                    sim_vals = pairwise_distances(self.vectors[indx], center[np.newaxis, :],
                                                  metric='euclidean').flatten()
                else:
                    sim_vals = pairwise_distances(vec[indx], center[np.newaxis, :], metric='euclidean').flatten()
                sort_inds = np.argsort(sim_vals)
            elif select_mtd == 'random':
                sort_inds = np.random.permutation(len(indx))

            # randomly sample some avoid names if it is too long
            if len(cluster_names) > 300:
                sample_ind = np.random.choice(len(cluster_names), 300, replace=False)

            # verbose or not
            if verbose and i % 10 == 0:
                show_prompt = True
            else:
                show_prompt = False

            summary_center, tokens = self.summary_contents(indx=indx,
                                                           sort_inds=sort_inds,
                                                           avoid_content=cluster_names,
                                                           max_texts=max_texts,
                                                           prompt_language=prompt_language,
                                                           num_topics=num_topics,
                                                           max_length=max_length,
                                                           show_prompt=show_prompt)
            all_tokens += tokens
            if verbose and i % 10 == 0:
                print(f"Raw Keywords for center {i}:", summary_center)
            processed_center, tokens = self.core_text_filter([summary_center], max_length)
            all_tokens += tokens
            if verbose and i % 10 == 0:
                print(f"Processed Keywords for center {i}:", ",".join(processed_center))
            cluster_names.extend(processed_center)
            cluster_names = process_strings(cluster_names)

        print('Before Post Process:', len(cluster_names))

        if post_process:
            cluster_names, tokens = self.core_text_filter(cluster_names, max_length)
            all_tokens += tokens

        print('After Post Process:', len(cluster_names))

        cluster_names = list(set(cluster_names))
        output_keywords = list(set(self.keywords or []) | set(cluster_names)) if add_keywords else cluster_names
        self.keywords = process_strings(output_keywords)
        self.keyvectors = np.array(self.embeddings.embed_documents(self.keywords))

        return cluster_names, all_tokens

    #################################### functions for knowledge graph construction
    def distance_core_seg(self, core_texts, core_labels=None, k=20,
                          dist_metric='cosine', method='annoy', return_full=False, return_prob=False):
        # consider to write coresearch into a subclass
        core_ebds = np.array(self.embeddings.embed_documents(core_texts))
        if core_labels is None:
            core_labels = np.arange(len(core_ebds))
        else:
            core_labels = np.array(core_labels)

        k = min(k, len(core_texts))
        if method == 'annoy':
            if dist_metric == 'cosine':
                similarity = 'angular'
            else:
                similarity = dist_metric
            knn_ind, knn_dist = autoKG.ANN_search(self.vectors, core_ebds, k, similarity=similarity)
        elif method == 'dense':
            dist_mat = pairwise_distances(self.vectors, core_ebds, metric=dist_metric)
            knn_ind = []
            knn_dist = []
            for i in range(len(dist_mat)):
                knn_ind.append(np.argsort(dist_mat[i])[:k])
                knn_dist.append(dist_mat[i][knn_ind[i]])
            knn_ind = np.array(knn_ind)
            knn_dist = np.arccos(1 - np.array(knn_dist))
        else:
            sys.exit('Invalid choice of method ' + dist_metric)

        knn_ind = autoKG.replace_labels(knn_ind, core_labels)

        if return_prob:
            D = knn_dist * knn_dist
            eps = D[:, k - 1]
            weights = np.exp(-4 * D / eps[:, None])
            prob = weights / np.sum(weights, axis=1)[:, np.newaxis]

        if return_full:
            if return_prob:
                return knn_ind, knn_dist, prob
            else:
                return knn_ind, knn_dist
        else:
            if return_prob:
                return knn_ind[:, 0], knn_dist[:, 0], prob[:, 0]
            else:
                return knn_ind[:, 0], knn_dist[:, 0]

    def laplace_diffusion(self, core_texts, trust_num=10, core_labels=None, k=20,
                          dist_metric='cosine', return_full=False):
        # we need to update the graph based rather than making a new one
        if self.weightmatrix is None:
            self.make_graph(k=20)

        knn_ind, knn_dist = self.distance_core_seg(core_texts, core_labels, k,
                                                   dist_metric, method='annoy', return_full=False, return_prob=False)

        if core_labels is None:
            core_labels = np.arange(len(core_texts))

        else:
            core_labels = np.array(core_labels)

        select_inds = np.array([], dtype=np.int64)
        select_labels = np.array([], dtype=np.int64)
        all_inds = np.arange(len(self.vectors))
        for i in range(len(core_texts)):
            select_ind = all_inds[knn_ind == i][np.argsort(knn_dist[knn_ind == i])[:trust_num]]
            select_inds = np.concatenate((select_inds, select_ind))
            select_labels = np.concatenate((select_labels, core_labels[i] * np.ones(len(select_ind))))

        model = gl.ssl.laplace(self.weightmatrix)
        print(select_inds)
        U = model._fit(select_inds, select_labels)

        if return_full:
            return U
        else:
            return np.argmax(U, axis=1)

    def PosNNeg_seg(self, core_text, trust_num=5, k=20, dist_metric='cosine', negative_multiplier=3, seg_mtd='laplace'):
        if self.weightmatrix is None:
            self.make_graph(k=20)
        knn_ind, knn_dist = self.distance_core_seg([core_text], [0], k,
                                                   dist_metric, method='dense', return_full=False, return_prob=False)
        sort_ind = np.argsort(knn_dist)
        select_inds = np.concatenate((sort_ind[:trust_num], sort_ind[-negative_multiplier * trust_num:]))
        select_labels = np.concatenate((np.zeros(trust_num), np.ones(negative_multiplier * trust_num)))
        if seg_mtd == 'kmeans':
            sub_core_texts = [self.texts[i] for i in select_inds]
            label_pred, U = self.distance_core_seg(sub_core_texts, select_labels, k,
                                                   dist_metric, method='dense', return_full=False, return_prob=False)
            U = np.exp(- U / np.max(U, axis=0))
        elif seg_mtd == 'laplace':
            model = gl.ssl.laplace(self.weightmatrix)
            U = model._fit(select_inds, select_labels)
            label_pred = np.argmax(U, axis=1)
            U = U[:, 0]
        elif seg_mtd == 'poisson':
            model = gl.ssl.poisson(self.weightmatrix)
            U = model._fit(select_inds, select_labels)
            label_pred = np.argmax(U, axis=1)
            U = U[:, 0]

        return label_pred, U


    def coretexts_seg_individual(self, trust_num=5, core_labels=None, k=20,
                                 dist_metric='cosine', negative_multiplier=3, seg_mtd='laplace',
                                 return_mat=True, connect_threshold=1):
        # we need to update the graph based rather than making a new one
        if self.weightmatrix is None:
            self.make_graph(k=20)

        core_texts = self.keywords
        if core_labels is None:
            core_labels = np.arange(len(core_texts))
        else:
            core_labels = np.array(core_labels)

        N_labels = np.max(core_labels) + 1
        U_mat = np.zeros((len(self.texts), len(core_labels)))
        pred_mat = np.zeros((len(self.texts), N_labels))

        for core_ind, core_text in enumerate(core_texts):
            label_pred, U = self.PosNNeg_seg(core_text, trust_num, k, dist_metric, negative_multiplier, seg_mtd)
            # print(f"The number of data clustered as core text {core_ind} is {np.sum(label_pred==0)}")

            U_mat[:, core_ind] = U
            pred_mat[label_pred == 0, core_labels[core_ind]] = 1

        if connect_threshold < 1:
            num_conn = np.sum(pred_mat, axis=0)
            N = len(self.texts)
            large_inds = np.where(num_conn > N*connect_threshold)[0]
            num_elements = int(N * connect_threshold)
            for l_ind in large_inds:
                threshold = np.partition(U_mat[:, l_ind], -num_elements)[-num_elements]
                pred_mat[:, l_ind] = np.where(U_mat[:, l_ind] >= threshold, 1, 0)

        if return_mat:
            # A = autoKG.create_sparse_matrix_A(pred_mat, np.array(core_labels))
            A = csr_matrix((pred_mat.T@pred_mat).astype(int))
            A = A - diags(A.diagonal())
            self.U_mat = U_mat
            self.pred_mat = pred_mat
            self.A = A
            return pred_mat, U_mat, A
        else:
            self.U_mat = U_mat
            self.pred_mat = pred_mat
            return pred_mat, U_mat


    #############################################
    def content_check(self, include_keygraph=True, auto_embedding=False):
        is_valid = True
        if self.keywords is None:
            print('Please set up keywords as self.keywords')
            is_valid = False
        if self.keyvectors is None:
            if auto_embedding:
                self.keyvectors = np.array(self.embeddings.embed_documents(self.keywords))
            else:
                print('Please set up keyword embedding vectors as self.keyvectors')
                is_valid = False
        if self.vectors is None:
            if auto_embedding:
                self.keyvectors = np.array(self.embeddings.embed_documents(self.texts))
            else:
                print('Please set up texts embedding vectors as self.vectors')
                is_valid = False
        if include_keygraph:
            if self.U_mat is None:
                print("Please calculate the U_mat using 'coretexts_seg_individual'.")
                is_valid = False
            if self.pred_mat is None:
                print("Please calculate the pred_mat using 'coretexts_seg_individual'.")
                is_valid = False
            if self.A is None:
                print("Please calculate the adjacency matrix A using 'coretexts_seg_individual'.")
                is_valid = False
        return is_valid

    def get_dist_mat(self):
        if not self.content_check(include_keygraph=True):
            raise ValueError('Missing Contents')

        self.dist_mat = np.arccos(1 - pairwise_distances(self.keyvectors, self.vectors, metric='cosine'))

    ################################ functions for search and chat
    def angular_search(self, query, k=5, search_mtd='pair_dist', search_with='texts'):
        if not self.content_check(include_keygraph=False):
            raise ValueError('Missing Contents')

        if isinstance(query, str):
            query_vec = np.array(self.embeddings.embed_documents([query]))
        elif isinstance(query, list):
            query_vec = np.array(self.embeddings.embed_documents(query))
        else:
            raise ValueError("query should be either string or list")

        if search_with == 'texts':
            s_vecs = self.vectors
        elif search_with == 'keywords':
            s_vecs = self.keyvectors
        else:
            raise ValueError("You should search with either 'texts' or 'keywords'.")

        if search_mtd == 'pair_dist':
            dist_mat = np.arccos(1 - pairwise_distances(query_vec, s_vecs, metric='cosine'))
            knn_ind = np.zeros((len(query_vec), k))
            knn_dist = np.zeros((len(query_vec), k))
            for i in range(len(knn_ind)):
                knn_ind[i] = np.argsort(dist_mat[i])[:k]
                knn_dist[i] = dist_mat[i, knn_ind[i].astype(int)]
        elif search_mtd == 'knn':
            knn_ind, knn_dist = autoKG.ANN_search(query_vec, s_vecs, k, similarity='angular')
        else:
            sys.exit('Invalid choice of method ' + search_mtd)

        return knn_ind.astype(int), knn_dist

    def keyword_related_text(self, keyind, k, use_u=True):
        if not self.content_check(include_keygraph=True):
            raise ValueError('Missing Contents')
        if use_u:
            text_ind = np.argsort(self.U_mat[:, keyind])[::-1][:k].astype(int).tolist()
        else:
            if self.dist_mat is None:
                raise ValueError("dist_mat is None")
            else:
                text_ind = np.argsort(self.dist_mat[keyind, :])[:k].astype(int).tolist()

        return text_ind

    def top_k_indices_sparse(self, row_index, k):
        row = self.A.getrow(row_index)
        non_zero_indices = row.nonzero()[1]
        if non_zero_indices.size < k:
            return non_zero_indices
        non_zero_values = np.array(row.data)
        top_k_indices = non_zero_indices[np.argpartition(non_zero_values, -k)[-k:]]

        return top_k_indices.astype(int).tolist()

#################################################################################################################
    ### main knowledge graph search function
    def KG_prompt(self, query, search_nums=(10, 5, 2, 3, 1), search_mtd='pair_dist', use_u=False):
        if not self.content_check(include_keygraph=True):
            raise ValueError('Missing Contents')

        text_ind = []
        keyword_ind = []
        adj_keyword_ind = []

        # search similar texts
        sim_text_ind, _ = self.angular_search(query, k=search_nums[0], search_mtd=search_mtd, search_with='texts')
        text_ind.extend([(i, -1) for i in sim_text_ind.tolist()[0]])

        # search similar keywords
        sim_keyword_ind, _ = self.angular_search(query, k=search_nums[1], search_mtd=search_mtd,
                                                 search_with='keywords')
        keyword_ind.extend(sim_keyword_ind.tolist()[0])
        for k_ind in sim_keyword_ind.tolist()[0]:
            t_ind = self.keyword_related_text(k_ind, k=search_nums[2], use_u=use_u)
            text_ind.extend([(i, k_ind) for i in t_ind])
            # adjacency search
            adj_text_inds = self.top_k_indices_sparse(k_ind, k=search_nums[3])
            adj_keyword_ind.extend([(i, k_ind) for i in adj_text_inds])
        adj_keyword_ind = remove_duplicates(adj_keyword_ind)
        adj_keyword_ind = [item for item in adj_keyword_ind if not item[0] in keyword_ind]

        # texts related to adjacency keywords
        for k_ind, _ in adj_keyword_ind:
            t_ind = self.keyword_related_text(k_ind, k=search_nums[4], use_u=use_u)
            text_ind.extend([(i, k_ind) for i in t_ind])

        text_ind = remove_duplicates(text_ind)

        record = {'query': query,
                  'text': text_ind,
                  'sim_keywords': keyword_ind,
                  'adj_keywords': adj_keyword_ind}

        return record

    def completion_from_record(self,
                               record,
                               output_tokens=1024,
                               prompt_language='English',
                               show_prompt=False,
                               prompt_keywords=True,
                               include_source=False):

        if self.llm_model.startswith("gpt-3.5"):
            model = "gpt-3.5-turbo-16k"
        else:
            model = self.llm_model

        if model.startswith("gpt-3.5"):
            max_num_tokens = 11900
        elif model.startswith("gpt-4"):
            max_num_tokens = 7900
        else:
            raise ValueError("Model should be either GPT-3.5 or GPT-4.")

        if prompt_keywords:
            header_part = """
You will be given a set of keywords directly related to a query, \
as well as adjacent keywords from the knowledge graph. \
Keywords will be separated by semicolons (;).

Relevant texts will be provided, enclosed within triple backticks. \
These texts contain information pertinent to the query and keywords.
"""
        else:
            header_part = """
Relevant texts will be provided, enclosed within triple backticks. \
These texts contain information pertinent to the query.
"""

        header = f"""
I want you to do a task, deal with a query or answer a question \
with some information from a knowledge graph.

{header_part}

Please note, you should not invent any information. Stick to the facts provided in the keywords and texts.

These additional data are meant to assist you in accurately completing the task. 
Your response should be written in {prompt_language}.

Avoid to show any personal information, like Name, Email, WhatsApp, Skype, and Website in your polished response.
"""
        max_content_token = max_num_tokens - output_tokens - 150

        query = record['query']
        text_ind = record['text']
        keyword_ind = record['sim_keywords']
        adj_keyword_ind = record['adj_keywords']

        keywords_info = "Keywords directly related to the query:\n\n" + "; ".join(
            [f"{self.keywords[i]}" for i in keyword_ind]) + "\n\n"
        keywords_info += "Adjacent keywords according to the knowledge graph:\n\n" + \
                         "; ".join([f"{self.keywords[i]}" for i, _ in adj_keyword_ind])

        chosen_texts = []
        chosen_texts_len = 0

        separator_len = len(self.encoding.encode(self.separator))
        if prompt_keywords:
            chosen_texts_len += len(self.encoding.encode(keywords_info))
        chosen_texts_len += len(self.encoding.encode(header))
        chosen_texts_len += len(self.encoding.encode(query))

        for t_ind, _ in text_ind:
            select_text = self.texts[t_ind]
            if include_source:
                select_source = self.source[t_ind]
                select_text = f"Source:{select_source}" + f"Content:{select_text}"

            num_tokens = len(self.encoding.encode(select_text))
            chosen_texts_len += num_tokens + separator_len
            if chosen_texts_len > max_content_token:
                break

            chosen_texts.append(self.separator + select_text)

        ref_info = "Selected reference texts:\n"
        ref_info += ''.join(chosen_texts)

        if prompt_keywords:
            prompt = f"""
{header}

{keywords_info}

Texts: 
'''{''.join(chosen_texts)}'''

Your task:
{query}

Your response:

"""
        else:
            prompt = f"""
{header}

Texts: 
'''{''.join(chosen_texts)}'''

Your task:
{query}

Your response:

"""

        if show_prompt:
            print(prompt)

        response, _, all_tokens = get_completion(prompt,
                                           model_name=model,
                                           max_tokens=output_tokens,
                                           temperature=self.temperature,
                                           top_p=self.top_p)


        return response, keywords_info, ref_info, all_tokens

#################################################################################################################

    ### graph visualization
    def draw_graph_from_record(self,
                               record,
                               node_colors=([0, 1, 1], [0, 1, 0.5], [1, 0.7, 0.75]),
                               node_shape='o',
                               edge_color='black',
                               edge_widths=(2, 0.5),
                               node_sizes=(500, 150, 50),
                               font_color='black',
                               font_size=6,
                               show_text=True,
                               save_fig=False,
                               save_path='Subgraph_vis.png'):

        T = record['text']
        K1 = record['sim_keywords']
        K2 = record['adj_keywords']
        N = [element.replace(" ", "\n") for element in self.keywords]
        Q = 'Query'

        G = nx.Graph()
        G.add_node(Q)

        for i in K1:
            G.add_edge(Q, N[i])

        for i in K1:
            for j in K1:
                if self.A[i, j] > 0:
                    G.add_edge(N[i], N[j])
            for k, _ in K2:
                if self.A[i, k] > 0:
                    G.add_edge(N[i], N[k])
        # for i, L in K2:
        #     for j in L:
        #         G.add_edge(N[i], N[j])

        if show_text:
            for i, L in T:
                new_node = f"Text {i}"
                G.add_node(new_node)
                for j in L:
                    if j == -1:
                        G.add_edge(new_node, Q)
                    else:
                        G.add_edge(new_node, N[j])

        color_map = {}
        color_map[Q] = node_colors[0]
        node_size_map = {node: node_sizes[0] for node in G.nodes}

        for node in N:
            color_map[node] = node_colors[1]
            node_size_map[node] = node_sizes[1]

        if show_text:
            for i, _ in T:
                color_map[f"Text {i}"] = node_colors[2]
                node_size_map[f"Text {i}"] = node_sizes[2]

        edge_width_map = {edge: edge_widths[0] for edge in G.edges}

        if show_text:
            for i, L in T:
                new_node = f"Text {i}"
                for j in L:
                    if j == -1:
                        edge_width_map[(new_node, Q)] = edge_widths[1]
                        edge_width_map[(Q, new_node)] = edge_widths[1]
                    else:
                        edge_width_map[(new_node, N[j])] = edge_widths[1]
                        edge_width_map[(N[j], new_node)] = edge_widths[1]

        # pos = nx.spring_layout(G, seed=42)
        pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50, scale=2.0)
        nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color=edge_color,
                               width=[edge_width_map[edge] for edge in G.edges])
        nx.draw_networkx_nodes(G, pos, node_color=[color_map[node] for node in G.nodes],
                               node_size=[node_size_map[node] for node in G.nodes], node_shape=node_shape)
        nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes}, font_size=font_size,
                                font_color=font_color)

        if save_fig:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
        plt.figure()

    ### save and load
    def save_data(self, save_path, include_texts=False):
        if include_texts:
            keywords_dic = {'keywords': self.keywords, 'keyvectors': self.keyvectors,
                            'U_mat': self.U_mat, 'pred_mat': self.pred_mat, 'A': self.A,
                            'texts': self.texts, 'embedding_vectors': self.vectors,
                            'dist_mat': self.dist_mat, 'token_counts': self.token_counts,
                            'source': self.source}
        else:
            keywords_dic = {'keywords': self.keywords, 'keyvectors': self.keyvectors,
                            'U_mat': self.U_mat, 'pred_mat': self.pred_mat, 'A': self.A,
                            'dist_mat': self.dist_mat}
        np.save(save_path, keywords_dic)
        print(f"Successfully save to {save_path}")

    def load_data(self, load_path, include_texts=False):
        keywords_dic = np.load(load_path, allow_pickle=True).item()
        self.keywords = keywords_dic.get('keywords')
        self.keyvectors = keywords_dic.get('keyvectors')
        self.U_mat = keywords_dic.get('U_mat')
        self.pred_mat = keywords_dic.get('pred_mat')
        self.A = keywords_dic.get('A')
        self.dist_mat = keywords_dic.get('dist_mat')
        if include_texts:
            if "texts" in keywords_dic:
                self.texts = keywords_dic.get('texts')
                self.vectors = keywords_dic.get('embedding_vectors')
                self.token_counts = keywords_dic.get('token_counts')
                self.source = keywords_dic.get('source')
            else:
                print("Fails to load texts information. Please check if your data includes a key named 'text'.")
        print(f"Successfully load from {load_path}")

    def write_keywords(self, save_path):
        if not self.content_check(include_keygraph=False):
            raise ValueError('Missing Contents')

        result = ''
        for i in range(len(self.keywords)):
            result += self.keywords[i]
            if (i + 1) % 10 == 0:
                result += '\n'
            else:
                result += '; '

        with open(save_path, 'w') as f:
            f.write(result)

    def check_completion(self):
        attributes_to_check = ['keywords', 'keyvectors', 'U_mat', 'pred_mat', 'A', 'dist_mat',
                               'texts', 'vectors', 'token_counts', 'source']

        for attr in attributes_to_check:
            if getattr(self, attr, None) is None:
                return False
        return True

    ## static methods
    @staticmethod
    def replace_labels(ind, labels):
        ind_new = np.zeros_like(ind)
        for i in range(len(labels)):
            ind_new[ind == i] = labels[i]
        return ind_new

    @staticmethod
    def ANN_search(X1, X2, k, similarity='euclidean'):
        # annoy search function
        # O(NlogN)
        M, d1 = X1.shape
        N, d2 = X2.shape

        assert d1 == d2, "The dimensions of datasets X1 and X2 do not match"

        if not similarity in ['euclidean', 'angular', 'manhattan', 'hamming', 'dot']:
            sys.exit('Invalid choice of similarity ' + similarity)

        d = d1
        k = min(k, X2.shape[0])

        t = AnnoyIndex(d, similarity)
        for i in range(N):
            t.add_item(i, X2[i])
        t.build(5)

        knn_dist = []
        knn_ind = []
        for x1 in X1:
            indices, distances = t.get_nns_by_vector(x1, k, include_distances=True)
            knn_ind.append(indices)
            knn_dist.append(distances)
        knn_ind = np.array(knn_ind)
        knn_dist = np.array(knn_dist)

        return knn_ind, knn_dist


