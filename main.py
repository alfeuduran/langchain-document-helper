from backend import core

import streamlit as st
from streamlit_chat import message

st.header("Langchain Document Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here")


# When initialize always be an empty list

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def create_sources_string(source_urls):
    if not source_urls:
        return ""
    sources_list = list(sources)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Geerating response..."):
        # generated_response = core.run_llm(prompt) agora adiciono todo o historico em uma nova chamada que antes só recebi aum parâmetro com isso teremos um novo tipo de chain onde vamos conseguir usar memory para recuperar o histórico do que foi falado.
        generated_response = core.run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = f"{generated_response['answer']} \n\n **Sources:** {create_sources_string(sources)}"

        print(formatted_response)

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, formatted_response))

if st.session_state["chat_answers_history"]:
    print(st.session_state["chat_answers_history"])
    # for i, (user_prompt, chat_answer) in enumerate(zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"])):
    #     st.write(f"**User Prompt {i+1}:** {user_prompt}")
    #     st.write(f"**Chat Answer {i+1}:** {chat_answer}")

    # test
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response, is_user=False)


if st.session_state["user_prompt_history"]:
    print(st.session_state["user_prompt_history"])


#  Adiconar depois a inicialização do Vectorstore quando iniciamos o app juntamente com o index existente que foi criado no momento da ingestão.
