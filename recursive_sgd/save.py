import pickle
def save(model, filename='model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    return

