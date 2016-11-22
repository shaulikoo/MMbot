import pickle


def save_var(var_list, file_full_path='.\objs.pickle'):
    try:
        # Saving the objects:
        with open(file_full_path, 'w') as f:
            pickle.dump(var_list, f)
        return True
    except Exception as e:
        print ("error in save_var")
        print (e.message)
        return False


def load_var(file_full_path='.\objs.pickle'):
    try:
        # Getting back the objects:
        with open(file_full_path) as f:
            var_list = pickle.load(f)
        return var_list
    except:
        return None


