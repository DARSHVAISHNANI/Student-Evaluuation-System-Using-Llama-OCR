from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re


class AnswerEvaluator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the AnswerEvaluator with a specific model.
        
        :param model_name: Name of the Hugging Face model to be used.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def set_model(self, model_name: str):
        """
        Update the model and tokenizer with a new model.
        
        :param model_name: Name of the new Hugging Face model to use.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, text: str) -> torch.Tensor:
        """
        Generate embeddings for a given text using the pre-trained model.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use mean pooling for embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def compute_similarity(self, original: str, student: str) -> float:
        """
        Compute cosine similarity between original and student answers.
        """
        original_emb = self.get_embeddings(original)
        student_emb = self.get_embeddings(student)
        return cosine_similarity(original_emb, student_emb)[0][0]

    def extract_key_points(self, reference_answer: str) -> list:
        """
        Extract key points from the reference answer by splitting sentences.
        """
        key_points = re.split(r'[.!?]', reference_answer)
        return [kp.strip() for kp in key_points if kp.strip()]

    def check_key_points_coverage(self, key_points: list, student_answer: str) -> dict:
        """
        Check how well the student's answer covers each key point.
        """
        coverage = {}
        for key_point in key_points:
            coverage[key_point] = self.compute_similarity(key_point, student_answer)
        return coverage

    def assign_marks_with_weighting(self, coverage: dict, total_marks: int) -> float:
        """
        Assign marks based on coverage with weighting.
        """
        total_weight = sum(coverage.values())
        weighted_marks = 0
        for key_point, score in coverage.items():
            weight = score / total_weight if total_weight > 0 else 1 / len(coverage)
            weighted_marks += total_marks * weight * (score if score > 0.5 else 0)
        return round(weighted_marks, 2)

    def calculate_penalties(self, student_answer: str, key_points: list) -> int:
        """
        Calculate penalties for the student's answer.
        """
        penalties = 0
        # Penalize irrelevance: based on length not matching key points
        if len(student_answer.split()) > len(" ".join(key_points).split()) * 1.5:
            penalties += 1  # Example penalty for verbosity
        # Add more penalty logic as needed
        return penalties

    def generate_feedback(self, coverage: dict) -> list:
        """
        Generate feedback based on the similarity scores of key points.
        """
        feedback = []
        for key_point, score in coverage.items():
            if score > 0.85:
                feedback.append(f"Good match for: '{key_point}'")
            elif score > 0.5:
                feedback.append(f"Partial match for: '{key_point}'")
            else:
                feedback.append(f"Missing or poor match for: '{key_point}'")
        return feedback

    def evaluate_answer(self, reference_answer: str, student_answer: str, total_marks: int) -> tuple:
        """
        Evaluate the student's answer against the reference answer.
        """
        # Step 1: Extract key points from the reference answer
        key_points = self.extract_key_points(reference_answer)

        # Step 2: Check coverage of key points
        coverage = self.check_key_points_coverage(key_points, student_answer)

        # Step 3: Assign marks with weighting
        marks = self.assign_marks_with_weighting(coverage, total_marks)

        # Step 4: Apply penalties
        penalties = self.calculate_penalties(student_answer, key_points)
        marks -= penalties
        marks = max(0, marks)  # Ensure non-negative marks

        # Step 5: Generate feedback
        feedback = self.generate_feedback(coverage)

        return round(marks), feedback
