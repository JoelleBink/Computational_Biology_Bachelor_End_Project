# Computational Biology Bachelor End Project
Bachelor End Project Joëlle Bink 2021 Computational Biology supervised by Francesca Grisoni

The paper corresponding to this GitHubm, titled “Evaluation of machine learning methods for bioactivity prediction in the presence of activity cliffs.” can be found here: [BEP_computational_biology_paper_activity_cliffs.pdf](https://github.com/JoelleBink/Computational_Biology_Bachelor_End_Project/files/6708842/BEP_computational_biology_paper_activity_cliffs.pdf)



**Data sets**
The data sets obtained in this paper are organised in the following way: {CHEMBLID}\_{train/test}\_{descriptor name}.csv, such that all 36 possible combinations of three ChEMBL datasets with six molecular descriptor types and independent training and test sets are saved as csv files individually.
An example showing the first five lines of {CHEMBL224\_test\_constitutional} can be found here: 

![tablehead_data](https://user-images.githubusercontent.com/45286571/123257181-1e45d800-d4f2-11eb-826b-7e74a5a36f2c.JPG)

The ''Y'' column contains experimental potency values, which are computed as logarithms of the affinity to the receptor in nanomolar units, where higher potency values mean higher affinity and activity. In every data set, the fourth until the last column contain the descriptors and corresponding values for all molecules, used as ''X'' input for the machine learning models. 
Every data set starts with the same three columns: "row index", "smiles" and "Y". 

**Code & file details**
All coding is provided with thorough commenting. An overview of what each code and file is used for can be found here: 

**File map**: **Paper visualisation**
This file map shows all Figures used in the paper “Evaluation of machine learning methods for bioactivity prediction in the presence of activity cliffs.”

**File map**: **descriptors_data**
This file map shows all data sets structured as described above.

**File map**: **Notebooks**
This file map shows all code needed to perform data scaling, machine learning and visualisation as described in the paper.

- **variables** stores all metric values in .csv files
- **images** stores all visualisations created in .png files
- **activity_cliffs_filtering** shows all variables and images for machine learning with filtering for activity cliffs.
- **machine_learning_script_withcliffs.ipynb** is the Python code used to perform machine learning without filtering for activity cliffs.
- **models_tables.ipynb** is the Python code that creates a merged table of all metric values for all combinations of data sets, models and descriptor types. 
- **activity_cliffs.py** is the Python code used for identification of activity cliffs, used in:
- **cliff_visualisation.ipynb** is the Python code that shows indices and visualisations of activity cliffs.
