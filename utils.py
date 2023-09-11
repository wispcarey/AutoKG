import os, re

import tiktoken
import string
import time
import openai
import traceback

from docx import Document
from PyPDF2 import PdfReader
import pandas as pd

def get_num_tokens(texts, model):
    num_tokens = []
    encoding = tiktoken.encoding_for_model(model)
    for text in texts:
        num_tokens.append(len(encoding.encode(text)))
    return num_tokens

def get_completion(prompt, model_name="gpt-3.5-turbo-16k", max_tokens=256, retry_times=3, temperature=0.1, top_p=0.3):
    for i in range(retry_times):
        try:
            t = time.time()
            if model_name.startswith("gpt-3.5") or model_name.startswith("gpt-4"):
                messages = [{"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                result = response.choices[0].message["content"]
            else:
                response = openai.Completion.create(
                    model=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                result = response.choices[0].text

            t_completion = time.time() - t
            # 回答，所用时间，所用token
            return result, t_completion, response.usage.total_tokens

        except Exception as e:
            if i < retry_times - 1:
                if model_name.startswith("gpt-4"):
                    time.sleep(25)
                else:
                    time.sleep(1)
            else:
                raise ValueError(f"Max retries exceeded. Error: {e}")

def process_strings(strings):
    def normalize(s):
        return ''.join(c.lower() for c in s if c not in string.whitespace and c not in string.punctuation)

    strings = [s.strip() for s in strings]

    unique_strings = []
    normalized_set = set()

    for s in strings:
        normalized_s = normalize(s)
        if normalized_s not in normalized_set:
            normalized_set.add(normalized_s)
            unique_strings.append(s)

    return unique_strings

def remove_duplicates(S):
    dic = {}
    for i, j in S:
        if i not in dic:
            dic[i] = [j]
        else:
            if j not in dic[i]:
                dic[i].append(j)

    no_duplicates = []
    for i, j in dic.items():
        no_duplicates.append((i, j))

    return no_duplicates

def clean_text(text):
    text = text.replace("///", " ").replace("_x000D_", " ").replace("、", " ")

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'\n\n+', '\n\n', text)

    text = text.strip()

    return text

def my_text_splitter(input_text, chunk_size=200, separator=None, model='gpt-3.5-turbo-16k'):
    if separator is None:
        separator = ["\n\n", ".", " "]

    if not separator:
        return []
    outputs = []
    sep = separator[0]
    seg_texts = input_text.split(sep)
    text_tokens = get_num_tokens(seg_texts, model)
    temp_tokens = 0
    for text, text_token in zip(seg_texts, text_tokens):
        if not text:
            continue
        if temp_tokens + text_token <= chunk_size:
            if temp_tokens == 0:
                outputs.append(text)
            else:
                outputs[-1] += sep + text
            temp_tokens += text_token
        else:
            if text_token > chunk_size:
                sub_output = my_text_splitter(text, chunk_size, separator[1:])
                outputs.extend(sub_output)
                temp_tokens = 0
            else:
                outputs.append(text)
                temp_tokens = text_token

    return outputs


def split_texts_with_source(text_list, source_list, chunk_size=200, separator=None):
    seg_texts = []
    new_sources = []
    for i, text in enumerate(text_list):
        try:
            temp_seg_texts = my_text_splitter(text, chunk_size=chunk_size, separator=separator)
            seg_texts.extend(temp_seg_texts)
            new_sources.extend([source_list[i] for _ in temp_seg_texts])
        except:
            raise ValueError(f"{i}" + traceback.format_exc())

    return seg_texts, new_sources

def find_files(directory, filetype='docx'):
    docx_files = []
    sub_paths = []

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(f".{filetype}"):
                docx_files.append(filename)
                sub_paths.append(dirpath)

    return docx_files, sub_paths

def load_and_process_files(directory,
                           chunk_size=200,
                           separator=None):
    '''
    Load all docx, pdf, and xlsx files under the directory
    Then segment them into text chunks
    '''
    if separator is None:
        separator = ["\n\n", "\n", ". ", " "]

    raw_texts = []
    raw_sources = []

    # docx
    docx_files, sub_paths = find_files(directory, filetype='docx')

    for docx_file, sub_path in zip(docx_files, sub_paths):
        try:
            doc = Document(os.path.join(sub_path, docx_file))
            raw_texts.append("\n".join([t.text for t in doc.paragraphs]))
            raw_sources.append(docx_file)
        except Exception as e:
            print("Failed to load the file:", sub_path, docx_file)
            print("Error message:", str(e))
            continue

    # pdf
    pdf_files, sub_paths = find_files(directory, filetype='pdf')

    for pdf_file, sub_path in zip(pdf_files, sub_paths):
        raw_sources.append(pdf_file)
        try:
            pdf_path = os.path.join(sub_path, pdf_file)
            with open(pdf_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()  # 使用新方法 extract_text
                raw_texts.append(text)
        except Exception as e:
            print("Failed to load the file:", sub_path, pdf_file)
            print("Error message:", str(e))
            continue

    # xlsx
    xlsx_files, sub_paths = find_files(directory, filetype='xlsx')

    for xlsx_file, sub_path in zip(xlsx_files, sub_paths):
        try:
            xlsx_path = os.path.join(sub_path, xlsx_file)
            df = pd.read_excel(xlsx_path)

            if 'text' in df.columns:
                raw_texts.extend(df['text'].tolist())

            if 'source' in df.columns:
                source_list = df['source'].tolist()
                raw_sources.extend([xlsx_file + '\n' + s for s in source_list])


        except Exception as e:
            print("Failed to load the file:", sub_path, xlsx_file)
            print("Error message:", str(e))
            continue

    # clean text
    processed_texts = [clean_text(text) for text in raw_texts]

    # segmentation
    texts, sources = split_texts_with_source(processed_texts,
                                             raw_sources,
                                             chunk_size=chunk_size,
                                             separator=separator)

    return texts, sources


