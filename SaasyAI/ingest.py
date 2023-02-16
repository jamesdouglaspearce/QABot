"""This scrip ingests .txt and .csv files into a vectorDB to be used by QABot"""

from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import argparse
import glob
import os
import pandas as pd

parser = argparse.ArgumentParser(description='Ingests data into vectorDB')
parser.add_argument('-f', '--file_path',  dest='path', type=str,
                    help='Path to .txt or .csv files. All files in path will be added.')
parser.add_argument('-c', '--columns',  dest='cols', type=str,
                    help='Comma separated list of column names to ingest (for .csv only).')
parser.add_argument('-o', '--output_path',  dest='output', type=str, default='.',
                    help='Path to put output artifacts (default is current dir).')
parser.add_argument('-k', '--openai_key',  dest='key', type=str, default='openai_key.txt',
                    help='Text file that contains key (default openai_key.txt)')
parser.add_argument('-v', '--verbose',  dest='verbose', type=bool, default=True,
                    help='To be verbose')


args = parser.parse_args()

# Reads key from openai_key.txt
os.environ["OPENAI_API_KEY"] = open(args.key).read()

# Here we load in the data in the two format .txt and .csv
text_files = glob.glob(f"{args.path}/*.txt")
csv_files = glob.glob(f"{args.path}/*.csv")

text_data = []
text_sources = []
for text_file in text_files:
    if args.verbose:
        print(f"Reading file: {text_file}")
    with open(text_file) as f:
        text_data.append(f.read())
    file_name = text_file.split('/')[-1].split('.')[0]
    text_sources.append(file_name)

# Here we split the text documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(text_data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": text_sources[i]}] * len(splits))

if args.verbose:
    print(f"Number of .txt chunks {len(docs)}")

# Here for the CSV files we chunk by row, only selecting columns in -c arg.
for csv_file in csv_files:
    if args.verbose:
        print(f"Reading file: {csv_file}")
    file_name = csv_file.split('/')[-1].split('.')[0]
    df = pd.read_csv(csv_file, usecols=args.cols.split(','))
    for i, row in df.iterrows():
        t = []
        for col in row.items():
            t.append(str(col[1]))
        docs.append('\n'.join(t))
        metadatas.append({"source:": file_name})

if args.verbose:
    print(f"Number of .txt and .csv chunks {len(docs)}")

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open(f"{args.output}/faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)