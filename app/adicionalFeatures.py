# import libraries
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Text Length Extractor for feature unions
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    Add the length of the text message as a feature to dataset
    
    The assumption is people who is in urgent disaster condition will prefer to use less words to express
    '''
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).applymap(len)