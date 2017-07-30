import json

def decode_predictions(pred):
    CLASS_INDEX = json.load(open('../saved_models/hand_class_index.json'))
    top_indices = list(range(len(pred)))
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    return result
