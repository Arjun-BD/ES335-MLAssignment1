# All the tasks have been divided into folders and have been worked upon individually. Each folder contains ipynb files where each question for every task has been answered.

The folder `Combined` has been made by running the script `CombineScript.py`, which contains raw accelerometer data of different subjects split into train and test folders. The data for each activity is stored in its respective named folders and contains CSV files by the subject.

The folder `DecisionTreeImplementation` contains 3 markdown files explaining the questions asked in the Decision Tree implementation section, which are aptly labeled as per question.

The folder `HAR` contains the UCI HAR Dataset, which is the featurised dataset we have been provided with. The Python file "MakeDataset.py" saves numpy arrays of the raw accelerometer data. These numpy arrays are `X_test.npy`, `X_train.npy`, `y_test.npy`, `y_train.npy`.

The folder `Graphs` contains images and graphs used in the markdown files.

The folder `Task1` contains the answers to Task 1 of the Assignment. Each of the questions is answered in aptly named files. Question 3 and Question 4 are answered in the same file, `Ass1ExploratoryDataAnalysisQ3Q4`.

The folder `task2` contains answers to Task2. All the subquestions are answered in the file. All answers are in the file `Ass1Task2.ipynb`.

The folder `Task3` contains task 3. The file `zeroshot-fewshot-new.ipynb` contains the answers to the questions, while the file `fewshot_scratch.ipynb` contains some experiments that we did to increase fewshot accuracy.

The folder `Task4` contains answers to Task 4 questions 1 and 2, where the decision tree has been implemented for the data we recorded.

The folder `Task4-prompt` contains answers to task 4 questions 3 and 4, where we have used few-shot prompting to predict the activities for the activities we recorded.

The folder `Task4-data` contains the recorded data and the preprocessing file for sampling the recorded data in the required frequency.  

The folder tree contains two files, `utils.py` and `base.py`. All the functions and metrics required to create a decision tree on our own are present here.

`auto-efficiency.py` contains code related to the problem Decision Tree Implementation Q2
`classification-exp.py` contains code related to the problem Decision Tree Implementation Q1
`experiments.py` contains code related to the problem Decision Tree Implementation Q3
