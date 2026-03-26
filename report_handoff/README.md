
# Instructions for whats in this folder

## code_construction:
these are the files that were used to create the project. any questions about the project or when creating the report, i would recommened
providing whatever LLM these 3 files. given that it will know evyerthing. With the files youll also know what the code does and how it was built. I'd personally avoid reading them and just give them to an LLM


## data set used:
https://github.com/GenImage-Dataset/GenImage?tab=readme-ov-file
this is reputable and used over things like kaggle as generators were explicitly split up and you could choose your own unlike the kaggle dataset if you search for one online.
Midjourney and BigGAN were the two chosen generators. 
Final project split (per generator ai/nature)
* train: 5,600 / 5,600
* val: 1,200 / 1,200
* test: 1,200 / 1,200
Current combined split counts (both generators together):

- `train.csv`: 22,400 images (11,200 per generator)
- `val.csv`: 4,800 images (2,400 per generator)
- `test.csv`: 4,800 images (2,400 per generator)

Please note that the model were trained on midjourney and biggan. this means they will not do well if given an ai picture from chatGPT or gemini for exmaple. THIS IS OK. this is a limitation of the system and if reported properly this shoudl not hurt the grade. 



## Research citations + course content master:

research citations is just a list of supporting thigns that were done before code constrution so the project would align with the report aftwewards if that makes sense. one of the papers is by the people who made the genimage dataset which coudl be importnat. 
course content master docuemnt is so the built project was aligned with course content. Not sure how useful that file will be at all, it was just used to align the project to somethign.



## Plots and metrics

in this folder there are plots metric for each of the runs baseline, cnn, evaluate, and predict.
tables and plots/graphs are made with Matplotlib

BASELINE:
metrics.json — main baseline results file; use the test metrics from here for the report comparison table.
summary.txt — quick plain-English summary of the baseline run and headline results.
config_used.yaml — exact settings used for the baseline run, for reproducibility.
confusion_matrix_test.png — shows how many real and AI images were classified correctly or incorrectly on the test set.
roc_curve_test.png — shows how well the baseline separates the two classes across thresholds; useful as a performance visual.


CNN:
metrics.json — main CNN results file; includes best validation metrics and final test metrics.
summary.txt — quick plain-English summary of the CNN training run and final results.
config_used.yaml — exact settings used for the CNN run, for reproducibility.
train_history.png — shows how training and validation performance changed over epochs.
confusion_matrix_test.png — shows how many real and AI images the CNN classified correctly or incorrectly on the test set.
roc_curve_test.png — shows how well the CNN separates the two classes across thresholds; useful as a performance visual.

EVALUATE:
metrics.json — main robustness evaluation results file; contains metrics for all tested conditions.
summary.txt — quick plain-English summary of the evaluation run across all conditions.
config_used.yaml — exact settings used for the robustness evaluation run.
evaluate_metrics_table.png — summary table of performance across clean, JPEG-compressed, and resized test conditions.
confusion_matrix_clean.png — shows clean test-set classification results.
confusion_matrix_jpeg_q40.png — shows classification results under a heavily compressed JPEG condition.


PREDICT:
predictions.csv — per-image inference output for the prediction run, including predicted class and confidence.
summary.txt — quick summary of the prediction run and how many inputs were processed.
config_used.yaml — exact settings used for the prediction run.



OTHER:
clean_model_comparison_table.png — side-by-side table comparing baseline and CNN test-set performance across the main metrics.
evaluate_metrics_table.png — side-by-side table comparing CNN robustness across clean, JPEG, and resize conditions.
