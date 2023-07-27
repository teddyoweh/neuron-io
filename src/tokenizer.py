from dataclasses import dataclass


class Tokenizer:
    tokens:dict ={}
    classes:set = set()
    def __init__(self):
        pass
    def fit(self,y:list):
        self.classes.update(set(y))
        self.tokens = {_:i for i,_ in enumerate(self.classes)}
    def tokenize(self,y:list):
        return [self.tokens[_] for _ in y]
    def fit_tokenize(self,y:list):
        self.fit(y)
        return self.tokenize(y)
    def inverse_transform(self, tokens: list):
        inv_tokens = {i: token for token, i in self.tokens.items()}
        return [inv_tokens[token] for token in tokens]
    
    

