import time

import groq
from groq import Groq


class GROQ_HANDLER:
    groq_api_list = [
        'key'
    ]

    def __init__(self, max_retries=5):
        self.client = None
        self.key_number = 0
        self.get_groq_client(self.key_number)
        self.max_retries = max_retries

    def get_groq_client(self, key_index):
        self.client = Groq(
            api_key=self.groq_api_list[key_index],
        )
        return self.client

    def request_groq(self, system_prompt, user_prompt=None, temperature=0.63, top_p=0.44):
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        if user_prompt is not None:
            messages.append({
                "role": "user",
                "content": user_prompt,
            })
        # try:
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            # model="llama3-70b-8192",
            temperature=temperature,
            top_p=top_p
        )
        output = chat_completion.choices[0].message.content
        return output

    def get_next_key(self):
        return (self.key_number + 1) % len(self.groq_api_list)

    def get_llm_response(self, system_message, user_message, temp=0.63, top_p=0.44, ):
        for i in range(self.max_retries):
            try:
                response = self.request_groq(system_message, user_message, temp, top_p)
                return response
            except groq.InternalServerError:
                print("Internal Server Error")
                self.key_number = self.get_next_key()
                self.get_groq_client(self.key_number)
            except groq.APIConnectionError as e:
                print("The server could not be reached")
                print(e.__cause__)  # an underlying Exception, likely raised within httpx.
                self.key_number = self.get_next_key()
                self.get_groq_client(self.key_number)
            except groq.RateLimitError as e:
                print("A 429 status code was received; we should back off a bit.")
                self.key_number = self.get_next_key()
                self.get_groq_client(self.key_number)
            except groq.APIStatusError as e:
                print("Another non-200-range status code was received")
                print(e.status_code)
                print(e.response)
                self.key_number = self.get_next_key()
                self.get_groq_client(self.key_number)


