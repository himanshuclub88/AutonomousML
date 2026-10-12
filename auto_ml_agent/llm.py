from __future__ import annotations

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .config import settings


def get_llm() -> ChatOpenAI:
    if not settings.llm_api_key:
        raise RuntimeError("Missing LLM_API_KEY. Add it to your project .env file.")
    if not settings.llm_model:
        raise RuntimeError("Missing LLM_MODEL. Add it to your project .env file.")
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        timeout=800,
        max_retries=2,
    )


def build_structured_chain(
    system_prompt: str,
    output_model: type[BaseModel],
) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=output_model)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nFollow these output instructions exactly:\n{format_instructions}"),
        ("human", "{input}"),
    ])
    return prompt.partial(
        format_instructions=parser.get_format_instructions()
    ) | get_llm() | parser


def generate_reply(
    *,
    system_prompt: str,
    messages: list[dict[str, str]],
    response_model: type[BaseModel] | None = None,
) -> str | BaseModel:
    conversation = "\n\n".join(
        f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')}"
        for msg in messages
    )

    if response_model is not None:
        chain = build_structured_chain(system_prompt=system_prompt, output_model=response_model)
        try:
            return chain.invoke({"input": conversation})
        except OutputParserException:
            if "reply" not in response_model.model_fields:
                raise
            raw = get_llm().invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=conversation),
            ])
            if not isinstance(raw.content, str):
                raise ValueError("LLM returned non-text content during fallback.")
            return response_model(reply=raw.content)

    raw = get_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=conversation),
    ])
    if not isinstance(raw.content, str):
        raise ValueError("LLM returned non-text content.")
    return raw.content
