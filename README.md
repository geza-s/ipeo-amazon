# Project on Understanding the Amazon from Space
Better monitoring and understanding of the dynamics in the Amazon rainforest is necessary to make informed decisions to manage and protect it.
In this project, you are tasked to perform multi-class classification of highresolution satellite images to identify different features on the ground, as well as atmospheric conditions, in the Amazon basin.
Data In total, there are 40479 images with ground-sample distance of 3.7m. They come from Planet’s Flock 2 satellites and were collected between January 1, 2016 and February 1, 2017. All of the scenes come from the Amazon basin which includes Brazil, Peru, Uruguay, Colombia, Venezuela, Guyana, Bolivia, and Ecuador. Only RGB bands are provided.
The class labels for this task represent a reasonable subset of phenomena of interest in the Amazon basin. The labels can broadly be broken into three groups: atmospheric conditions, common land cover/land use phenomena, and rare land cover/land use phenomena. Each image will have one and potentially more than one atmospheric label and zero or more common and rare labels. The labels are provided in a .csv file where every row contains labels for given image, separated by space.

## Example of label : 
a) ”clear”, ”habitation”, ”road”, ”bare ground” (b) ”clear”, ”water”, ”primary” (c) ”partly cloudy”, ”primary”

# Challenges
 Training examples may (and mostly do) have more than one label. Therefore, for each image, you need to decide on each label whether the corresponding feature is present in the image or not. Also, when you split the dataset into training, validation, and test dataset you need to ensure that each dataset contains enough examples of each class. This is complicated by the fact that the classes are unevenly distributed.