class Reflection():
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, chat_history):
        last_question = chat_history[-1]["content"]
        prompt = f"Hãy diễn đạt lại câu hỏi sau sao cho rõ ràng hơn trong ngữ cảnh sau đây:\n\n"
        for turn in chat_history[:-1]:
            prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
        prompt += f"\nUser (hiện tại): {last_question}\n\nCâu hỏi được diễn đạt lại:"

        response = self.llm.chat([{"role": "user", "content": prompt}])
        return response.strip()

