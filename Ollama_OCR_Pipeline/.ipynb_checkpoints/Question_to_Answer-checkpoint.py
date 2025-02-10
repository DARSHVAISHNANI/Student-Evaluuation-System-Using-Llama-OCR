from transformers import pipeline

class Question_to_Answer_ChatModel:
    def __init__(self, model_name="distilgpt2"):
        """
        Initialize the Question_to_Answer_ChatModel with a specified model name.
        :param model_name: Name of the Hugging Face model to load (default: distilgpt2).
        """
        self.model_name = model_name
        try:
            self.generator = pipeline("text-generation", model=self.model_name)
        except Exception as e:
            raise ValueError(f"Error loading model '{self.model_name}': {e}")

    def generate_answer(self, question, max_length=100):
        """
        Generate an answer to the given question using the model.
        :param question: The input question.
        :param max_length: Maximum length of the generated response.
        :return: Generated response as a string.
        """
        response = self.generator(question, max_length=max_length, num_return_sequences=1,truncation=True)
        return response[0]['generated_text']
