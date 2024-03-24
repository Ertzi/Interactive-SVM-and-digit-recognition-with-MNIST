# Interactive Digit Recognition

The aim of the proyect is firstly, to visualize different SVM models with different samples created in 2D by the user. Secondly, to apply SVM classifiers on the MNIST database and create an interactive "handwritten" digit recognition interface. The app also has a section to visualice some instances from the MNIST database.

Some comments:
- The script ["main.py"](main.py) needs the databases on the execution path file. Since databases can't be uploaded to GitHub, i've uploaded them to [Google Drive](https://drive.google.com/drive/folders/1gO1JQnIuk_lpRx5cP9Gho-lCodyaPakz?usp=sharing).
- In order to avoid large execution times, i've already trained a model for the MNIST database. The model too needs to be on the execution path file (along with the scripts). You can download the model [here](https://drive.google.com/drive/folders/13OXYSQ6K5MJ3fEXjDBzMSZ9RcwJ8jtKn?usp=sharing). If you don't want to download the model and prefer to train a model by yourselve, you can delete line 55 from "main.py" and uncomment lines 59-64, however this takes about two minutes of execution time before u can do anything.
- The models are trained on the lastest version of scikit-learn. If you run another version of Scikit-Learn, upgrade the package (with "py -m pip install --upgrade scikit-learn" in terminal") or train the model by yourselve with the ["generate_models.py"](generate_models.py) and use your model.
