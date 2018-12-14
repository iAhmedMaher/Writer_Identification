import os
import os.path 
import model
import time
import Utilities

F_results = open("results.txt","w")
F_time = open("time.txt","w")

def run_iteration(iteration_path, clf, feature_method='LPQ'):
    global F_results
    test_data = [os.path.join(iteration_path,'test.PNG')]
    writers_data = []
    iteration_files = os.listdir(iteration_path)
    for writer in iteration_files:
        writer_path = os.path.join(iteration_path,writer)
        current_writer = []

        if os.path.isdir(writer_path):
            for file in os.listdir(writer_path):
                current_writer.append(os.path.join(writer_path,file))
        
            writers_data.append(current_writer)   

    output = model.run_trial(writers_data,  test_data, clf, feature_method)
    print(output)
    F_results.write(str(output[0][0]+1))
    F_results.write("\n")


path = r'data'

files = os.listdir(path)

for file in files:
    full_path = os.path.join(path,file)
    if os.path.isdir(full_path):
        start = time.process_time()
        run_iteration(full_path, Utilities.map_str_to_clf('kNN'), 'LBP;LPQ')
        end = time.process_time()
        F_time.write(str(round(end-start, 2)))
        F_time.write("\n")
        #break

F_results.close()
F_time.close()