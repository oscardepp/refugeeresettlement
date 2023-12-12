# Refugee Resettlement Assistance Model (RRAM) 
<!-- <img width="1088" alt="modelcomparison" src="https://github.com/oscardepp/refugeeresettlement/assets/137336589/6478280b-574f-4ef2-8e23-76449577adf9"> -->
<p align="center">
  <img width="1088" alt="GMM Model Comparison" src="https://github.com/oscardepp/refugeeresettlement/assets/137336589/6478280b-574f-4ef2-8e23-76449577adf9">
</p>

<h5 align="center">Fig 1. GMM Model Comparison of 2016 Informal Settlements in Lebanon with UNHCR WASH Survey in 2013.</h5>

Refugee Resettlement ML Project 

For the full model shown in OpenStreetMaps in Lebanon, please visit: https://raw.githack.com/oscardepp/refugeeresettlement/main/intensity_map.html. 

## Introduction
This project assists refugee resettlement given a large-scale disaster in the world in a two-part, comprehensive model. The first part asks: a natural disaster hits a particular country: can we predict how many people will be displaced from their homes? In which the second part follows: given a proportion of those who are displaced, what are suitable locations to which they relocate in that country? Our model uses natural disasters and settlements in Lebanon to assess the damage, but the features and methodology could be easily generalized and abstracted to construct a more ubiquitous and robust framework.

## Data 

The data used across informal settlements are in Lebanon, provided by the Lebanese branch of the United Nations Office for the Coordination of Humanitarian Affairs(OCHA)[1]. The features found in this survey are universal features found in other surveys developed by the WHO and UNICEF like WASH used in Asian and African camps, assessing access to a protected/treated drinking water source, adequate quantities of water, use of toilets and latrines, and access to soap.
There were 6272 observations in this dataset. As unsupervised learning was used for grouping camps, the entire dataset was used for training. The 12 features include Status, Number of Tents, Number of Individuals, Number of Latrines, Water Capacity(Liters), Number of Substandard Buildings(SSBs), Number of Individuals living in SSBs, Type of Water Source Waste Disposal, Waste Water Disposal, Type of Internet Connection, and Free Vaccination for Children under 12.

## Data Processing



## Model

Gaussian Mixture Models and K-means clustering were used to generate groups of camps based on how adequate the features described in the data section were. The validity of the GMM was assessed by determining the closeness between the norms of the top ten Gaussians and Clusters using the Pearson Correlation Coefficient.

## Data Access & Citations

Dataset used: 
[1] Lebanon Inter-Agency OCHA, “Informal Settlements (Refugees living in informal settlements).” Lebanon, Dec. 31, 2016. 



