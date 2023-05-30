#!/usr/bin/env python

from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredEPubLoader

epub = 'process_book.epub'

loader = UnstructuredEPubLoader(epub)
docs = loader.load_and_split()

# print(docs)
# print(len(docs))

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

summary_chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    # return_intermediate_steps=True,
    # verbose=True,
)
# summary = summary_chain.run(docs)

template = """
以下の文章を日本語にしてください。
{text}
"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["text"],
)

ja_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    # return_intermediate_steps=True,
    # verbose=True,
)

ja_summary_chain = SimpleSequentialChain(
    chains=[summary_chain, ja_chain],
    # return_intermediate_steps=True,
    # verbose=True
)

summary = ja_summary_chain.run(docs)

print(summary)
