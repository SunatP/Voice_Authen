import os 
import pickle
import glob

# deletes a registered user from database
def delete_user():
    name = input("Enter name of the user:")

    [os.remove(path) for path in glob.glob('./voice_database/' + name + '/*')]
    os.removedirs('./voice_database/' + name)
    os.remove('./gmm_models/' + name + '.gmm')
    print("Remove Successfully")
    

if __name__ == '__main__':
    delete_user()