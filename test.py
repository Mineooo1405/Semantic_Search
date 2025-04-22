from Tool.OIE import extract_triples, extract_triples_for_search

text = "Barack Obama was born in Hawaii. Bill Gates founded Microsoft."
triples = extract_triples_for_search(text)

for t in triples:
    print(f"{t['subject']} | {t['relation']} | {t['object']}")