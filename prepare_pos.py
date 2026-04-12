"""
Generate word_pos.json: maps every word in glove_10k.json to its primary
part of speech  (noun / verb / adj),  using NLTK WordNet.
Words whose dominant POS is adverb, preposition, etc. are omitted.

Run once:
    pip install nltk
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
    python prepare_pos.py
"""
import json
from collections import Counter
from nltk.corpus import wordnet as wn

POS_LABEL = {
    wn.NOUN:    'noun',
    wn.VERB:    'verb',
    wn.ADJ:     'adj',
    wn.ADJ_SAT: 'adj',   # WordNet satellite adjectives (e.g. "best")
}

with open('glove_10k.json', encoding='utf-8') as f:
    vocab = json.load(f)

pos_map = {}
for word in vocab:
    synsets = wn.synsets(word)
    if not synsets:
        continue
    counts  = Counter(s.pos() for s in synsets)
    top_pos = counts.most_common(1)[0][0]
    if top_pos in POS_LABEL:
        pos_map[word] = POS_LABEL[top_pos]

print(f"Tagged {len(pos_map):,} / {len(vocab):,} words with noun / verb / adj.")
with open('word_pos.json', 'w', encoding='utf-8') as f:
    json.dump(pos_map, f)
print("Saved word_pos.json")
