from typing import List, Text
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message
from underthesea import word_tokenize
class VietnameseTokenizer(Tokenizer):
    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)
        text = text.lower()
        words = word_tokenize(text)
        return self._convert_words_to_tokens(words, text)
