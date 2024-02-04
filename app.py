import streamlit as st
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain_community.llms import CTransformers
from dotenv import load_dotenv

load_dotenv()


def get_hf_response(input_text, num_of_words, blog_style):
    llm = HuggingFaceHub(
        repo_id="meta-llama/Llama-2-7b-chat-hf",
        model_kwargs={"temperature": 0.01, "max_new_tokens": 256}
    )

    template = f"""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {num_of_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "num_of_words"],
                            template=template)

    response = llm(
        prompt.format(
            blog_style=blog_style,
            input_text=input_text,
            num_of_words=num_of_words
        )
    )

    print(response)
    return response


def get_llama_response(input_text, num_of_words, blog_style):

    llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type="llama",
                        config={
                            'max_new_tokens': 256,
                            'temperature': 0.01
                        })

    template = f"""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {num_of_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "num_of_words"],
                            template=template)

    response = llm(
        prompt.format(
            blog_style=blog_style,
            input_text=input_text,
            num_of_words=num_of_words
        )
    )

    print(response)
    return response


st.set_page_config(page_title="Generate Blogs",
                   page_icon="🤖",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    num_of_words = st.text_input("Number of Words")

with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'),
                              index=0)

submit = st.button("Generate")

if submit:
    st.write(get_hf_response(input_text, num_of_words, blog_style))
