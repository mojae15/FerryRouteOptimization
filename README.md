# Ferry_Route_Optimization

This is the code base for the Ferry Route Optimzation Project. Some files have not been pushed to this directory, as they were for testing purposes and such. All the models have not been pushed to this directory, however the data and code for building them are.

To run the route planning, go to the "RoutePlanning" folder, and run the "calc_route.sh" file, with either 1 or 2 as input.
    ./calc_route.sh 1
For routes going from Spodsbjerg to Tårs
    ./calc_route 2
For Routes going from Tårs to Spodsbjerg

The output will be a list of latitude and longitude coordinates, along with a speed.
Additionally, a .hmtl file called "path.html" file will be created, which contains a visual representation of the route.
Be sure to have run "make" in the "RoutePlanning" folder before running the program.

## Data Folder
The "Data" folder contains the data used for the project, as well as some programs used to extract data, filter data, and build datasets.
The folder also contains the datasets for the different data points.

### All Datasets Folder
The "All Datasets" folder contain all the datasets used for the project. The "clean_data.py" program separates the datasets from the different dates, into datasets for the different datapoints.

### Grid Folder
The "Grid" folder contains the code for building the grid used in the routeplanning. This is the file "make_grid.py", which creates the grid. 
By changing the variable "grain" the grain of the grid can be changed.

### Relevant_Data Folder
The "Relevant_Data" folder contains the programs used to build the dataset we use for the models. The different programs in this folder extracts the different information we want, which mainly consists of extracting data from the passage mode. The "build_dataset.py" program takes all the .csv files in the folder and builds a single .csv file from them, which is used for bulding the models.

## Models
The "Models" folder contains the code used for bulding the various models used in the project. 
"network.py" is for creating all neural networks with the different parameters we've set (WARNING: This takes a while)
"network_single.py" is for creating a single neural network.
"R_Random_Forest.R" is for creating all the random forests with the different parameters we've set-

## RoutePlanning
The "RoutePlanning" folder contains the code for doing the actual route planning, mainly the "RoutePlanning.cpp" file, which is the program for calcualting the route.
Additionally, the folder also contains the "build_dataset.py" program, which builds the dataset needed for the route planning. 
If building the dataset is not desired, as this takes some time since it gets its data from a online api, the line in "calc_route.sh" where this program is called can be commented out, or "./rp" can just be run seperately.

## Table Data Folder
The "Table Data" folder contains datasets which contain information about the current at various dates. The "build_dataset.py" program converts the values in these datasets to singular dataset for the datapoints needed.