# Iterating through dataset to extract the original sentence and its translation 
def get_all_sentences(ds, lang):
    for pair in ds:
        yield pair['translation'][lang]