from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

MODEL_NAME = "gpt-4"

prompt = PromptTemplate.from_template("""
Please correct any unnatural English in the following conversation.
And then, explain in Japanese how unnatural it is.

===
{conversation}
===
""")


def chat_invoke(messages):
    chat_model = ChatOpenAI(model=MODEL_NAME)
    return chat_model.invoke(messages)


if __name__ == "__main__":
    conversation = """
    A: What kinds of music do you like?
    B: I likes classic music.
    """

    messages = [HumanMessage(content=prompt.format(conversation=conversation))]

    print(chat_invoke(messages))
