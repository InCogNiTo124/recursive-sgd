import pickle
def save(model, filename='model.sgd'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    return

def load(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
