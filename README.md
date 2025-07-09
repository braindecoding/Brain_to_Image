# Brain to Image

This repository is a collection of scripts and code used in my Master's project. 

### Abstract 
This study explores the integration of Electroencephalography (EEG) data and Generative Adversarial Networks (GANs) to generate visual representations of perceived images. Utilizing the Mind Big Data (MBD) dataset and Emotiv EPOC® device, EEG signals were captured and pre-processed to remove noise and artifacts. A Convolutional Neural Network (CNN) was employed to classify and encode these signals into latent space vectors. These vectors, along with class conditioning labels, were fed into an Auxiliary Classifier GAN (ACGAN) to generate images. The study faced challenges such as signal complexity, noise, and limited training data. Despite these, the ACGAN model, particularly with a modulation layer and concatenation with embedding, demonstrated the ability to produce images with a Structural Similarity Index (SSIM) score of approximately 0.3, indicating a 65% similarity to the ground truth. The results validate the hypothesis that EEG signals can be used to generate representative images, although further improvements in data preprocessing and classification accuracy are needed. Future work could explore reinforcement learning to enhance the model's performance.

## Paper published BIOSIGNALS 2025, 18th International conference on Bio-Inspired system and signal porcessing
https://biosignals.scitevents.org/

## Datasets
All data used for training has been based the that provided by https://vivancos.com/ https://mindbigdata.com/opendb/index.html
In particular the data from https://huggingface.co/datasets/DavidVivancos/MindBigData2022, using dataset https://huggingface.co/datasets/DavidVivancos/MindBigData2022_MNIST_EP

Use raw_to_pickle to extract the eeg signal data from the raw files and create a pandas data table to make it easier to process.

## Filtering data

use filter_data to apply notch and bandwidth filters to data

## Use MNE to remove artifacts

use mne_pipeline to apply artifact removal, mne pipeline can also be used to apply filtering and display and inspect eeg data. https://mne.tools/stable/index.html

## Apply common average reference

use car_pipeline to apply common average reference (CAR), filter on correlation, apply sliding window and save train/test split data for training models.
The project paper uses method 2 where; 
  method 1 for CAR which is where the baselined car is subtracted from each signal and then the resultand signal is correlated with the CAR and selection is made > ~0.2 correlation
  method 2 for CAR which is where the correlation between the baselined car and each signal is taken and the selection is made for > ~0.9 correlation
in notebooks the pipeline_data not book shows how this is done with some comments.

## Training models

apply filtering, artifact removal and CAR, save test/train split as pickle file. Use the train scripts to laod the data and train GAN.


## Preprosesing

Pipeline Preprocessing EEG Data
1. Raw Data Extraction (raw_to_pickle.py)
  * Mengkonversi file CSV besar dari dataset MindBigData ke format pandas DataFrame
  * Menyimpan dalam format pickle untuk memudahkan pemrosesan selanjutnya
  * Dataset yang digunakan: MindBigData2022_MNIST_EP dengan 14 channel EEG
    ```python
    ## Takes the large raw data file given from MBD and creates a pandas datatable for easy usage, savig the DF to pickle file.

    big_df = batch.batch_process_csv_pandas(f"{root_dir}/{input_file}",f"{root_dir}/{output_file}")
    ```
2. Signal Filtering (filter_data.py)
Menerapkan dua jenis filter:
  * Notch Filter: Menghilangkan line noise (50/60 Hz)
  * Butterworth Bandpass Filter: Memfilter frekuensi dalam rentang tertentu
    ```python
    filtered = hf.apply_notch_filter(row[keys_to_import],sample_rate,notch_freqs=notch_freqs,notch_widths=notch_widths)
    filtered = hf.apply_butterworth_filter(filtered,sample_rate,lowcut,highcut,order=butter_order)
    ```
3. Artifact Removal (mne_pipeline.py)
Menggunakan library MNE untuk:
  * Menghilangkan artifact dari gerakan mata dan otot
  * Menerapkan epoch rejection untuk menghilangkan data yang buruk
  * Menggunakan Average Reference
  * Membuat fixed-length epochs (2 detik)
    ```python
    raw.filter(l_freq=lowcut, h_freq=highcut,verbose=False)
    raw.set_eeg_reference(ref_channels='average',ch_type='eeg',projection=False,verbose=False)
    epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True,verbose=False)
    epochs_clean = epochs.drop_bad(reject={'eeg': 100e-0},verbose=False)
    ```
4. Common Average Reference (CAR) (car_pipeline.py)
Implementasi dua metode CAR:
  * Method 1: CAR baseline dikurangi dari setiap sinyal, kemudian korelasi > 0.2
  * Method 2: Korelasi langsung antara CAR baseline dan setiap sinyal > 0.9 (digunakan dalam paper)
    ```python
    # Calculate ERP
    car = np.mean(arr, axis=0)
    # Baseline correction (using first 125 ms as baseline)
    baseline = np.mean(car[:16])
    car_corrected = car - baseline
    ```
5. Sliding Window Processing
  * Menerapkan sliding window dengan ukuran 32 sample dan overlap 4
  * Menghasilkan multiple windows dari setiap sinyal 2 detik
    ```python
    def sliding_window_eeg(signal, window_size=32, overlap=4):
        # Calculate the step size
        step = window_size - overlap
        # Calculate the number of windows
        num_windows = (len(signal) - window_size) // step + 1
    ```
6. Data Splitting dan Persiapan Training
  * Membagi data menjadi train/test split (90%/10%)
  * Mengkonversi label ke format categorical
  * Menyimpan dalam format yang siap untuk training GAN

Urutan Preprocessing Lengkap:
1. Raw CSV → Pickle (raw_to_pickle.py)
2. Apply Filters (filter_data.py) - Notch + Butterworth
3. Artifact Removal (mne_pipeline.py) - MNE pipeline
4. CAR Processing (car_pipeline.py) - Common Average Reference
5. Sliding Window - Membuat multiple windows per sinyal
6. Train/Test Split - Persiapan data untuk training

Parameter Kunci:
* Sampling Rate: 128 Hz
* Signal Length: 256 samples (2 detik)
* Channels: 14 EEG channels (AF3, AF4, F7, F8, F3, F4, FC5, FC6, T7, T8, P7, P8, O1, O2)
* Window Size: 32 samples dengan overlap 4
* CAR Correlation Threshold: 0.925 (method 2)