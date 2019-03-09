from enum import Enum

class EnumFeatureCombination(Enum):
    TEXTUAL = 'Textual'
    CATEGORICAL = 'Categorical'
    COOCCURENCE = 'Co-Occurence'
    IMAGE = 'Image'
    TEXTUAL_CATEGORICAL = 'Textual and Categorical'
    TEXTUAL_CATEGORICAL_COOCCURENCE = 'Textual, Categorical and Co-Occurence'
    TEXTUAL_COOCCURENCE = 'Textual and Co-Occurence'
    CATEGORICAL_COOCCURENCE = 'Categorical and Co-Occurence'
    TEXTUAL_COOCCURENCE_IMAGE = 'Textual, Co-Occurence and Image'
    TEXTUAL_IMAGE = 'Textual and Image'
    CATEGORICAL_COOCCURENCE_IMAGE = 'Categorical, Co-Occurence and Image'
    CATEGORICAL_IMAGE = 'Categorical and Image'
    COOCCURENCE_IMAGE = 'Co-Occurence and Image'
    TEXTUAL_CATEGORICAL_IMAGE = 'Textual, Categorical and Image'
    TEXTUAL_CATEGORICAL_COOCCURENCE_IMAGE = 'Textual, Categorical, Co-Occurence and Image'

