"""Python file to serve as the frontend"""
import faiss
from langchain import OpenAI, VectorDBQA, LLMChain, PromptTemplate
import pickle
from QABot import QABot
import os
import argparse

parser = argparse.ArgumentParser(description='Chat with SaasyAI')
parser.add_argument('-t', '--template_file',  dest='template', type=str, default='saasyai_template.txt',
                    help='Path to .txt template file (default saasyai_template.txt).')
parser.add_argument('-k', '--openai_key',  dest='key', type=str, default='openai_key.txt',
                    help='Text file that contains key (default openai_key.txt)')
parser.add_argument('-s', '--vector_store',  dest='store', type=str, default='faiss_store.pkl',
                    help='VectorDB store from ingest.py (default faiss_store.pkl)')
parser.add_argument('-i', '--index',  dest='index', type=str, default='docs.index',
                    help='Index from ingest.py (default docs.index)')
parser.add_argument('-T', '--temp',  dest='temp', type=float, default=0.0,
                    help='Temperature of OpenAI model (default 0.0)')
parser.add_argument('-m', '--model_name',  dest='model_name', type=str, default='text-davinci-003',
                    help='OpenAI model to use (default text-davinci-003)')
parser.add_argument('-v', '--verbose',  dest='verbose', type=bool, default=False,
                    help='To be verbose or not (default False)')

args = parser.parse_args()

os.environ["OPENAI_API_KEY"] = open(args.key).read()

# Load the LangChain.
index = faiss.read_index(args.index)

with open(args.store, "rb") as f:
    store = pickle.load(f)

store.index = index

model = OpenAI(temperature=args.temp, model_name=args.model_name, max_tokens=1000)

prompt = PromptTemplate(
    input_variables=["query", "source_documents", "chat_history"],
    template=open(args.template).read()
)
chain_qa = LLMChain(llm=model,
                    prompt=prompt,
                    verbose=args.verbose)

saasyai = QABot(chain_qa, store)

print("""                                                     
 ,---.                   ,---.           ,---.  ,--. 
'   .-'  ,--,--. ,--,--.'   .-',--. ,--./  O  \ |  | 
`.  `-. ' ,-.  |' ,-.  |`.  `-. \  '  /|  .-.  ||  | 
.-'    |\ '-'  |\ '-'  |.-'    | \   ' |  | |  ||  | 
`-----'  `--`--' `--`--'`-----'.-'  /  `--' `--'`--' 
                               `---'                 
=====================================================
Type 'exit' to quit. \n\n""")
print(f"SaaSyAI:{saasyai.query('hello who are you?')}")
print()
while True:
    q = input()
    print()
    if q == 'exit':
        break
    print(f"SaaSyAI:{saasyai.query(q)}\n" )