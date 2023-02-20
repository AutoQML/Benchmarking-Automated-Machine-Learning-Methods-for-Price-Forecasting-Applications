# Machine-Price-Prediction with AutoQML
The following project examins different methods and algorithms for predicting the price for used construction machines.  
The overall goal of the project is to automate the Machine-Learning part as much as possible.  
As a definition the ML-part within this project starts by reading in the cleaned and for ML purposses prepared (scaled and one-hot-encoded) data.  

The whole **ML pipeline** will be defined as follows:

1. Gathering the data for price prediction. Tha data must contain the following features: 
	1. Price of the used construction machine
	2. Construction year 
	3. Number of working hours
	4. Country / location
	5. Machine model extensions  

	The goal of this step is to collect as many datapoints as possible for different construction machine models.


2. Evaluating the date with the help of domain knowledge experts.
	1. What kind of outliers are there within the data?
	2. Are the features equaly important or are there features that are more important and should be given a higher weight? Weighing the features can be done within a range of **0 to 1**. The value **0.5** is neutral. Feature importance is increasing from **0.5 to 1** and decreasing from **0.5 to 0**.
	3. Are there *subfeatures* within the **extension** feature. How important are these subfeatures? 
	4. Are there features or subfeatures that are from special interrest independent of their frequency of occurrence? Eg. is there a rare extension that makes the construction machine particularly expensiv? This feature has to bekome a high weight becaus it has a spacial importance for the price of the machine.  
	
	The goal of this step is to understand the data with the help of the data domain experts and to create an explainable data set.  
	
3. Prepare the data for ML
	1. **Detect outliers**.
	2. **Detect redundant data points** which can be totally identical or alter in at most one feature value. For example the same construction machine is offerd with different, increasing working hours. Therefore the values of the working hours are different, but it is the same machine. In such a case the most up to date data point has to be choosen and all the other datapoints have to be discarded. This step makes sure, that each construction machine will be considerd just once. 
	3. **One-hot encode** the categorical attributes *location* and *extension*.
	4. **Scale** all numerical attributes.   