from enum import Enum
import os
from dotenv import load_dotenv

import instructor
import google.generativeai as genai
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()


class TicketCategory(str, Enum):
    GENERAL = "general"
    ORDER = "order"
    BILLING = "billing"
    OTHER = "other"


class UserTicketReply(BaseModel):
    """A model for responding to users and storing them as tickets."""

    content: str = Field(description="The reply sent to the customer.")
    category: TicketCategory = Field(description="The type of ticket.")

    model_config = ConfigDict(use_enum_values=True)


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="gemini-1.5-flash",
    ),
    mode=instructor.Mode.GEMINI_JSON,
)

query = "Hi there, I have a question about my bill. Can you help me?"

messages = [
    {
        "role": "system",
        "content": "You're a helpful customer care assistant. Always end your messages with the category of the incoming message.",
    },
    {
        "role": "user",
        "content": query,
    },
]


async def extract() -> UserTicketReply:
    return await client.messages.create(
        messages=messages,
        response_model=UserTicketReply,
    )


response: UserTicketReply = extract()
