from editdistance import distance

def calc_cer(target_text, predicted_text) -> float:
    dist = distance(list(target_text), list(predicted_text))
    length = len(target_text)
    if length != 0:
        cer = dist / length
    else: cer = dist


def calc_wer(target_text, predicted_text) -> float:
    target_text, predicted_text = target_text.split(), predicted_text.split()
    dist = distance(target_text, predicted_text)
    length = len(target_text)
    if length != 0:
        cer = dist / length
    else: cer = dist