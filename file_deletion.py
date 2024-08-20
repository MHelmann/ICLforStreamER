import os

def remove_files(file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' removed.")
        else:
            print(f"File '{file_path}' does not exist.")


if __name__ == "__main__":
        
    files_to_remove = ["/home/helmanmm/code/pools/train.out", 
                       "/home/helmanmm/code/logs/pool_logs.log",
                       "/home/helmanmm/code/output/dl_errors.csv", 
                       "/home/helmanmm/code/output/train_error.csv",
                       "/home/helmanmm/code/models/model_pool/model.pt",
                       "/home/helmanmm/code/models/model1/model.pt",
                       "/home/helmanmm/code/models/model2/model.pt",
                       "/home/helmanmm/code/classifier/app1.log",
                       "/home/helmanmm/code/classifier/app2.log",
                       "/home/helmanmm/code/classifier/app1_PID.txt",
                       "/home/helmanmm/code/classifier/app2_PID.txt"]
    remove_files(files_to_remove)
