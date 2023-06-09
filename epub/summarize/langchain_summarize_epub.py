#!/usr/bin/env python

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredEPubLoader

epub = 'process_book.epub'

loader = UnstructuredEPubLoader(epub)
docs = loader.load_and_split()

# print(docs)
# print(len(docs))

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

map_template = """あなたはプロのライターで、内容を網羅しながらも分かりやすく要約することが得意です。
以下は書籍の内容の一部です。1500文字以内で要約してください。:

########
{text}
########

要約:"""

combine_template = """あなたはプロのライターで、内容を網羅しながらも分かりやすく要約することが得意です。
1つの書籍を分割して、分割した内容に対して、それぞれ要約した結果を以下に順に並べています。この書籍を1000~1500文字で要約してください。:

########
{text}
########

書籍の要約:"""

map_prompt = PromptTemplate(
    template=map_template,
    input_variables=["text"],
)

combine_prompt = PromptTemplate(
    template=combine_template,
    input_variables=["text"],
)

ja_summary_chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    # verbose=True,
)

ja_summary = ja_summary_chain.run(docs)

print(ja_summary)
