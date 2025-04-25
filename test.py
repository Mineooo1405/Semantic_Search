from Tool.Enhanced_Sentence_Detector import detect_sentences

text = "'We're not ever going to feed the whole city this way,' cautions Hardy. 'But if enough unused space can be developed like this, there's no reason why you shouldn't eventually target maybe between 5% and 10% of consumption.'"

sentences = detect_sentences(text)
for s in sentences:
    print(s)