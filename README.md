# tidio-nlp-task

## data
Folder for keeping the data.
- conversations.csv 
-  ratings.csv

<b> All the data should be kept on private repositories. In order to use the code, data should be copied here. </b>

## notebooks
Folder for jupyter notebooks.
- **Tidio NLP Task** - main notebook

<b> All functions developed in notebooks which may be used outside of those notebooks 
should be saved to utilities.</b>

## utilities
Folder with all the shared code as scripts <i>.py</i>.
- utils.py

## standalone files

- requirements.txt
- README.md (this file)
- .gitignore


## install dependencies

Using ``conda``:
```shell
$ conda create -n tidio python=3.7.3
$ conda activate tidio
```

In folder repository:
```shell
$ pip install -r requirements.txt
```
Install kernel for jupyter:
```shell
$ pip install --user ipykernel
$ python -m ipykernel install --user --name=tidio
$ jupyter notebook
```
And we are ready for exploration.
