from typing import Dict
import streamlit as st
import wikipedia
from transformers import Pipeline
from transformers import pipeline

NUM_SENT = 10
model_name = "deepset/roberta-base-squad2"

@st.cache
def get_qa_pipeline() -> Pipeline:
    qa = pipeline("question-answering")
#    qa = pipeline("question-answering", model=model_name, tokenizer=model_name)
    return qa


def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> Dict:
    input = {
        "question": question,
        "context": paragraph
    }
    return pipeline(input)

#####################

@st.cache
def get_wiki_paragraph(query: str) -> str:
    results = wikipedia.search(query)
    try:
        summary = wikipedia.summary(results[0], sentences=NUM_SENT)
    except wikipedia.DisambiguationError as e:
        ambiguous_terms = e.options
        return wikipedia.summary(ambiguous_terms[0], sentences=NUM_SENT)
    return summary


def format_text(paragraph: str, start_idx: int, end_idx: int) -> str:
    return paragraph[:start_idx] + "**" + paragraph[start_idx:end_idx] + "**" + paragraph[end_idx:]

#####################

if __name__ == "__main__":
    """
    # Wikipedia Article
    """
    paragraph_slot = st.empty()
    wiki_query = st.text_input("WIKIPEDIA SEARCH TERM", "")
    question = st.text_input("QUESTION", "")

    if wiki_query:
        wiki_para = get_wiki_paragraph(wiki_query)
        paragraph_slot.markdown(wiki_para)
        # Execute question against paragraph
        if question != "":
            pipeline = get_qa_pipeline()
            st.write(pipeline.model)
            st.write(pipeline.model.config)
            try:
                answer = answer_question(pipeline, question, wiki_para)

                start_idx = answer["start"]
                end_idx = answer["end"]
                paragraph_slot.markdown(format_text(wiki_para, start_idx, end_idx))
            except:
                st.write("You must provide a valid wikipedia paragraph")  
  








# import streamlit as st
# from transformers import pipeline

# title = 'Toy Question Answering App v0.1'
# context = "Rock Paper scissors can be played between two or more people. To begin, choose your opponent and stand or sit across from them.Both players close one hand into a fist and shake it while counting down. While counting down, players say, 'Rock, Paper, Scissors, Shoot' or 'Three, Two, One, Shoot!' on Shoot, everyone makes one of the three different hand signals/To make a rock, close your hand into a fist. To make paper, hold your hand out flat, with your palm facing downward.To make scissors, hold out your first two fingers in a V-shape. In Rock Paper Scissors, rock beats scissors by crushing them. Scissors beat paper by cutting the paper in two. paper beats rock by covering the paper. Players can play one game, best out of three games or as many games as they want!"
# question1 = "How many players are there?"
# question2 = "Does Rock beat paper?"
# question3 = "Do scissors beat paper?"

# model_name = "deepset/roberta-base-squad2"

# question_answerer = pipeline("question-answering", model=model_name, tokenizer=model_name)


# NUM_SENT = 10

# @st.cache
# def get_qa_pipeline() -> Pipeline:
#     qa = pipeline("question-answering", model=model_name, tokenizer=model_name)
#     return qa


# def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> Dict:
#     input = {
#         "question": title,
#         "context": context
#     }
#     return pipeline(input)
 
# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.title("Ask question about this video")
# st.text("What would you like to know today?")

# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# # with st.spinner ('Loading Model into Memory....'):
# #     retriever = get_retriever()
# #     reader = get_reader()
# #     pipe = ExtractiveQAPipeline(reader, retriever)  
# text = st.text_input('Enter your questions here....')
# if text:
#     st.write("Response:")
#     with st.spinner('Searching for answers....'):
#         prediction = pipe.run(query=text, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 2}})
#         st.write('answer: {}'.format(prediction['answers'][0].answer))
#         st.write('answer: {}'.format(prediction['answers'][1].answer))
#     st.write("")
