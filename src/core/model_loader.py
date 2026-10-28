# models/model_loader.py
from transformers import pipeline
import spacy
from spacy.cli import download
import torch

from .retrieval import RetrievalEngine

class ModelLoader:

    # MoritzLaurer/deberta-v3-large-zeroshot-v2.0
    # MoritzLaurer/ModernBERT-large-zeroshot-v2.0
    # qa_model_name="bert-large-uncased-whole-word-masking-finetuned-squad"
    # qa_model_name="deepset/deberta-v3-large-squad2"

    def __init__(self, nli_model_name="MoritzLaurer/ModernBERT-large-zeroshot-v2.0", 
                       qa_model_name="ahotrod/deberta-v3-large-finetuned-squadv2", 
                       spacy_model="en_core_web_sm", index_path='', device=0):
        self.nli_model_name = nli_model_name
        self.qa_model_name = qa_model_name
        self.spacy_model = spacy_model
        self.index_path = index_path
        self.device = device
        self.classifier = self.load_classifier()
        self.qa_model = self.load_qa_model()
        self.nlp = self.load_spacy_model()
        self.retrieval_engine = self.load_retrieval_engine()
        print("\nClassifier and NLP model loaded successfully.\n")

    def load_intent_dict(self, intent_dict):
        self.intent_slot_dict = intent_dict

    def load_classifier(self):
        print(f"Loading zero-shot classifier: {self.nli_model_name}...")
        return pipeline("zero-shot-classification", model=self.nli_model_name, device=self.device)

    def load_qa_model(self):
        print(f"Loading question-answering model: {self.qa_model_name}...")
        return pipeline("question-answering", model=self.qa_model_name, device=self.device)

    def load_spacy_model(self):
        print(f"Loading spaCy model: {self.spacy_model}...")
        try:
            return spacy.load(self.spacy_model)
        except OSError:
            print(f"Modelo '{self.spacy_model}' não encontrado. Instalando...")
            download(self.spacy_model)
            return spacy.load(self.spacy_model)
    
    def load_retrieval_engine(self):
        engine = RetrievalEngine(index_path=self.index_path, top_k=100, model="PL2")
        return engine
