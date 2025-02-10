import re
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class ResponseEvaluator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.sbert_model = SentenceTransformer(model_name)
    
    @staticmethod
    def preprocess_text(text):
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.lower())).strip()
    
    def calculate_sbert_similarity(self, ref, cand):
        ref_embedding = self.sbert_model.encode([ref])[0]
        cand_embedding = self.sbert_model.encode([cand])[0]
        similarity = np.dot(ref_embedding, cand_embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(cand_embedding))
        return similarity

    @staticmethod
    def calculate_bleu_similarity(ref, cand):
        return sentence_bleu([ref.split()], cand.split(), smoothing_function=SmoothingFunction().method1)
    
    @staticmethod
    def calculate_redundancy_penalty(text):
        words = text.split()
        unique_words = set(words)
        redundancy_ratio = 1 - len(unique_words) / len(words) if words else 0
        return max(0, redundancy_ratio - 0.2)  # Allow 20% redundancy
    
    def evaluate_response(self, student_answer, evaluator_answer, weights):
        student_answer = self.preprocess_text(student_answer)
        evaluator_answer = self.preprocess_text(evaluator_answer)
        
        bleu = self.calculate_bleu_similarity(evaluator_answer, student_answer)
        sbert_similarity = self.calculate_sbert_similarity(evaluator_answer, student_answer)
        redundancy_penalty = self.calculate_redundancy_penalty(student_answer)
        
        scores = {
            "BLEU Score": bleu * 100,
            "SBERT Similarity": sbert_similarity * 100,
            "Redundancy Penalty": redundancy_penalty * 100
        }
        
        return scores

    def calculate_marks(self, scores, total_marks=10, weights=None):
        if weights is None:
            weights = {
                "BLEU Score": 0.5,
                "SBERT Similarity": 0.5,
                "Redundancy Penalty": -0.1
            }
        
        final_score = sum(scores[aspect] * weight for aspect, weight in weights.items())
        marks_obtained = max(0, (final_score / 100) * total_marks)
        return round(marks_obtained, 2)

    def generate_feedback(self, scores, marks_obtained, total_marks, thresholds=None):
        if thresholds is None:
            thresholds = {
                "SBERT Similarity": 70,
                "Redundancy Penalty": 10
            }
        
        feedback = [
            f"Content Similarity (BLEU): {scores['BLEU Score']:.2f}%",
            f"Semantic Similarity (SBERT): {scores['SBERT Similarity']:.2f}%",
            f"Redundancy Penalty: -{scores['Redundancy Penalty']:.2f}%",
            f"Marks: {round(marks_obtained)}/{total_marks}"
        ]
        
        if scores["SBERT Similarity"] < thresholds["SBERT Similarity"]:
            feedback.append("Suggestion: Improve semantic alignment with the evaluator's answer.")
        if scores["Redundancy Penalty"] > thresholds["Redundancy Penalty"]:
            feedback.append("Suggestion: Avoid repeating phrases unnecessarily.")
        
        return "\n".join(feedback)
