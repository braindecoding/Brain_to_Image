from Brain_to_Image import batch_csv as batch

root_dir = "Datasets/MindBigData - The Visual MNIST of Brain Digits/2022Data"

input_file = "train_MindBigData2022_MNIST_EP.csv"
output_file = "train_MindBigData2022_MNIST_EP.pkl"
## TEST
# input_file = "test_MindBigData2022_MNIST_EP.csv"
# output_file = "test_MindBigData2022_MNIST_EP.pkl"
## Takes the large raw data file given from MBD and creates a pandas datatable for easy usage, savig the DF to pickle file.

big_df = batch.batch_process_csv_pandas(f"{root_dir}/{input_file}",f"{root_dir}/{output_file}")
big_df.info()
