import os
import re
import time
import string
import traceback
from typing import List, Tuple, Optional, Dict, Any

import yaml
import tiktoken

from openai import OpenAI

from docx import Document
from PyPDF2 import PdfReader
import pandas as pd


# -----------------------------
# Config loading (safe)
# -----------------------------
_DEFAULT_CONFIG_PATH = "config.yaml"


def _safe_load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError:
            return {}


_PARAMS = _safe_load_yaml(_DEFAULT_CONFIG_PATH)

# Environment variables take precedence over config.yaml
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or _PARAMS.get("OPENAI_API_KEY")
COMPLETIONS_MODEL = os.getenv("OPENAI_API_MODEL") or _PARAMS.get("OPENAI_API_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or _PARAMS.get("EMBEDDING_MODEL", "text-embedding-3-small")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -----------------------------
# Token utils
# -----------------------------
def get_num_tokens(texts: List[str], model: str) -> List[int]:
    # Token counting with a robust fallback for unknown model names.
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens: List[int] = []
    for text in texts:
        num_tokens.append(len(encoding.encode(text or "")))
    return num_tokens


# -----------------------------
# OpenAI completion (v1 SDK)
# -----------------------------
def get_completion(
    prompt: str,
    model_name: Optional[str] = None,
    max_tokens: int = 250,
    retry_times: int = 3,
    temperature: float = 0.9,
    top_p: float = 1.0,
    system: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Tuple[str, float, int]:
    """
    Generate chat completions using OpenAI v1 SDK.

    Returns:
        (response_text, time_taken_seconds, total_tokens_used)
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Put it in env var or config.yaml.")

    model = model_name or COMPLETIONS_MODEL
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=timeout_s)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err = None
    for attempt in range(retry_times):
        try:
            start_time = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            end_time = time.time()

            text = resp.choices[0].message.content or ""
            tokens_used = 0
            if getattr(resp, "usage", None) is not None and resp.usage is not None:
                tokens_used = int(resp.usage.total_tokens or 0)

            return text, (end_time - start_time), tokens_used

        except Exception as e:
            last_err = e
            # Exponential backoff with cap
            sleep_s = min(2 ** attempt, 20)
            time.sleep(sleep_s)

    raise ValueError(f"Failed to generate completion after retries. Last error: {last_err}")


# -----------------------------
# String helpers
# -----------------------------
def process_strings(strings: List[str]) -> List[str]:
    def normalize(s: str) -> str:
        return "".join(c.lower() for c in s if c not in string.whitespace and c not in string.punctuation)

    strings = [s.strip() for s in strings if s is not None]

    unique_strings: List[str] = []
    normalized_set = set()

    for s in strings:
        if not s:
            continue
        normalized_s = normalize(s)
        if normalized_s not in normalized_set:
            normalized_set.add(normalized_s)
            unique_strings.append(s)

    return unique_strings


def remove_duplicates(S: List[Tuple[int, int]]) -> List[Tuple[int, List[int]]]:
    dic: Dict[int, List[int]] = {}
    for i, j in S:
        if i not in dic:
            dic[i] = [j]
        else:
            if j not in dic[i]:
                dic[i].append(j)

    no_duplicates: List[Tuple[int, List[int]]] = []
    for i, j in dic.items():
        no_duplicates.append((i, j))

    return no_duplicates


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("///", " ").replace("_x000D_", " ").replace("、", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\n+", "\n\n", text)
    return text.strip()


# -----------------------------
# Text splitting
# -----------------------------
def my_text_splitter(
    input_text: str,
    chunk_size: int = 200,
    separator: Optional[List[str]] = None,
    model: str = "gpt-3.5-turbo-16k",
) -> List[str]:
    if input_text is None:
        return []

    if separator is None:
        separator = ["\n\n", ".", " "]

    if not separator:
        return []

    outputs: List[str] = []
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
                sub_output = my_text_splitter(text, chunk_size, separator[1:], model=model)
                outputs.extend(sub_output)
                temp_tokens = 0
            else:
                outputs.append(text)
                temp_tokens = text_token

    return outputs


def split_texts_with_source(
    text_list: List[str],
    source_list: List[str],
    chunk_size: int = 200,
    separator: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    seg_texts: List[str] = []
    new_sources: List[str] = []
    for i, text in enumerate(text_list):
        try:
            temp_seg_texts = my_text_splitter(text, chunk_size=chunk_size, separator=separator)
            seg_texts.extend(temp_seg_texts)
            new_sources.extend([source_list[i] for _ in temp_seg_texts])
        except Exception:
            raise ValueError(f"{i}" + traceback.format_exc())

    return seg_texts, new_sources


# -----------------------------
# File loading
# -----------------------------
def find_files(directory: str, filetype: str = "docx") -> Tuple[List[str], List[str]]:
    files: List[str] = []
    sub_paths: List[str] = []

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(f".{filetype}"):
                files.append(filename)
                sub_paths.append(dirpath)

    return files, sub_paths


def load_and_process_files(
    directory: str,
    chunk_size: int = 200,
    separator: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Load all docx, pdf, and xlsx files under the directory,
    then segment them into text chunks.
    """
    if separator is None:
        separator = ["\n\n", "\n", ". ", " "]

    raw_texts: List[str] = []
    raw_sources: List[str] = []

    # docx
    docx_files, sub_paths = find_files(directory, filetype="docx")
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
    pdf_files, sub_paths = find_files(directory, filetype="pdf")
    for pdf_file, sub_path in zip(pdf_files, sub_paths):
        raw_sources.append(pdf_file)
        try:
            pdf_path = os.path.join(sub_path, pdf_file)
            with open(pdf_path, "rb") as f:
                pdf_reader = PdfReader(f)
                text_parts: List[str] = []
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                raw_texts.append("\n".join(text_parts))
        except Exception as e:
            print("Failed to load the file:", sub_path, pdf_file)
            print("Error message:", str(e))
            continue

    # xlsx
    xlsx_files, sub_paths = find_files(directory, filetype="xlsx")
    for xlsx_file, sub_path in zip(xlsx_files, sub_paths):
        try:
            xlsx_path = os.path.join(sub_path, xlsx_file)
            df = pd.read_excel(xlsx_path)

            if "text" in df.columns:
                raw_texts.extend([str(x) if x is not None else "" for x in df["text"].tolist()])

            if "source" in df.columns:
                source_list = df["source"].tolist()
                raw_sources.extend([xlsx_file + "\n" + str(s) for s in source_list])

        except Exception as e:
            print("Failed to load the file:", sub_path, xlsx_file)
            print("Error message:", str(e))
            continue

    processed_texts = [clean_text(text) for text in raw_texts]

    texts, sources = split_texts_with_source(
        processed_texts,
        raw_sources,
        chunk_size=chunk_size,
        separator=separator,
    )

    return texts, sources
