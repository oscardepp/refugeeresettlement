# Refugee Resettlement Assistance Model (RRAM) 
<!-- <img width="1088" alt="modelcomparison" src="https://github.com/oscardepp/refugeeresettlement/assets/137336589/6478280b-574f-4ef2-8e23-76449577adf9"> -->
<p align="center">
  <img width="1088" alt="GMM Model Comparison" src="https://github.com/oscardepp/refugeeresettlement/assets/137336589/6478280b-574f-4ef2-8e23-76449577adf9">
</p>

<h5 align="center">Fig 1. GMM Model Comparison of 2016 Informal Settlements in Lebanon with UNHCR WASH Survey in 2013[1].</h5>

Refugee Resettlement ML Project 

For the full model shown in OpenStreetMaps in Lebanon, please visit: https://raw.githack.com/oscardepp/refugeeresettlement/main/intensity_map.html. 

## Introduction
This project assists refugee resettlement given a large-scale disaster in the world in a two-part, comprehensive model. The first part asks: a natural disaster hits a particular country: can we predict how many people will be displaced from their homes? In which the second part follows: given a proportion of those who are displaced, what are suitable locations to which they relocate in that country? Our model uses natural disasters and settlements in Lebanon to assess the damage, but the features and methodology could be easily generalized and abstracted to construct a more ubiquitous and robust framework. Having refugees settle based on quality of life and not just proximity seems like a better metric to use. 

## Data 

The data used across informal settlements are in Lebanon, provided by the Lebanese branch of the United Nations Office for the Coordination of Humanitarian Affairs(OCHA)[2]. The features found in this survey are universal features found in other surveys developed by the WHO and UNICEF like WASH used in Asian and African camps, assessing access to a protected/treated drinking water source, adequate quantities of water, use of toilets and latrines, and access to soap.
There were 6272 observations in this dataset. As unsupervised learning was used for grouping camps, the entire dataset was used for training. The 12 features include Status, Number of Tents, Number of Individuals, Number of Latrines, Water Capacity(Liters), Number of Substandard Buildings(SSBs), Number of Individuals living in SSBs, Type of Water Source Waste Disposal, Waste Water Disposal, Type of Internet Connection, and Free Vaccination for Children under 12.

## Data Processing

<p align="center">
  <img width="879" alt="Image" src="https://github.com/oscardepp/refugeeresettlement/assets/137336589/ac9b739c-2112-4138-8326-6ac46f9c370a">
</p>
<h5 align="center">Fig 2. Rankings of Water Sources according to WASH survey guidelines[3].</h5>

Qualitative Features were converted into quantitative rankings loosely following guidelines published by the UNHCR's Water, Sanitation & Hygiene (WASH) survey & Standardized Expanded Nutrition Survey(SENS), with an example mapping of water sources to numbers shown in Figure 2. Some features had over 50% missing or "other" data, as this information was taken from the UN and in refugee camps where often information about latrines, water capacity, or individuals living in an area is unknown. We treated the large-scale unknown data as a missing category, with a neutral feature. Some disadvantages of this are that this increases the variance and uncertainty of the model quite significantly, but allows statements about the informal settlements. 
<p align="center">
<img width="642" alt="image" src="https://github.com/oscardepp/refugeeresettlement/assets/137336589/c8c0c5cb-f198-4de7-b56f-3e9d759b2f64">
</p>
<h5 align="center">Fig 3. Converted Quantitative Scales from Categorical Features.</h5>

These were then scaled using sklearn's MinMaxScaler to weigh each numerical and categorical feature equally from 0 to 1. 

## Model

Gaussian Mixture Models, unsupervised learning techniques, were used to group informal settlements by the twelve features used. The means of the top ten Gaussians were compared against the centers of each cluster in K-means clustering, and the number of Gaussians and Clusters was determined by how similar they were(evaluated using the Pearson Correlation Coefficient). To have the GMM be independent of the K-means clustering, the GMM in Sklearn was initialized to 'random'. Scores, or the ranking of each group of camps were determined by calculating the norm of each Gaussian mean. The scores were then plotted with the latitude and longitude pairs on OpenStreetMaps(OSM).

## Results 

Figures 4-6 illustrate some ways that these scores can be utilized for practical means. 

| ![Image 1](https://github.com/oscardepp/refugeeresettlement/assets/137336589/ac6ff8d5-63b9-46e7-8a9a-2e6007f7e84f) | ![Image 2](https://github.com/oscardepp/refugeeresettlement/assets/137336589/28a09870-1bb7-44b8-b735-c8d7bcb97d2d) |
| --- | --- |
| **Fig 4a. Aakar Model Comparison.** | **Fig 4b. Hermel Model Comparison.** |
<p align="center">
<img width="1045" alt="image" src="https://github.com/oscardepp/refugeeresettlement/assets/137336589/38384d12-dffb-481e-9fe1-39da7b11c976">
</p>
<h5 align="center">Fig 5. Combining Education metrics with WASH survey data to make conclusions about quality of life.</h5>
<p align="center">
<img width="1070" alt="image" src="https://github.com/oscardepp/refugeeresettlement/assets/137336589/6e8d51c0-b0a5-4bf6-b38f-6f601157b5fd">
</p>
<h5 align="center">Fig 6. Best regions according to our model and other surveys to move to if needed temporarily.</h5>

## Analysis

From Figures 4a and 4b, it is clear that in the Hermel and Aakar regions, this model of informal settlements is consistent with the survey results from 2013. However, looking at a region like Beirut, we see that there is no data for informal settlements there because it is an urban setting.  Our model provides more specificity than the WASH survey results in this diagram provide, as our model demonstrates the mean of each grouping–demonstrating how it is satisfying these requirements, as well as for the bottom Gaussians and how it could improve.  

This model could easily be combined with other information like educational data shown in Figures 5 and 6. With access to UNHCR branches and public schools nearby, Zahle provides a good haven for people displaced to move to, or if the Lebanese government chooses to work on those locations, which are optimal starting places.  Figure 6 demonstrates clusters of "good locations" marked in green and indicates the best places to relocate to for large groups of people.  

Possible extensions of this project include combining it quantitatively with education data, as well as using vulnerability data to assess how much conflict there is in these areas. This model is easily scalable, as the same criteria are used in African and South Asian countries where similar establishments and settlements exist. 



## Data Access & Citations

Dataset used: 

[1] F. Coloni, D. Adams, and F. Hanga, “Syria refugee response,” Syria Refugee Response WASH Sector Working Group Implementation of Water and Sanitation Activities January to July 2013, https://www.ecoi.net/en/file/local/1144452/1930_1377696781_syria-refugee-response.pdf (accessed Dec. 12, 2023). <br>
[2] Lebanon Inter-Agency OCHA, “Informal Settlements (Refugees living in informal settlements).” Lebanon, Dec. 31, 2016. <br>
[3] UNHCR, UNHCR Standardised Expanded Nutrition Survey (SENS) Guidelines for Refugee Populations Module 7:Water, Sanitation, &amp; Hygiene (WASH). UNHCR, 2018 



