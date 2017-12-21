#GTD Denovo Vesrion.
Input: 
	1. 3 bams (parents bams and kid's bam.)
	2. List of location of de novo variants to be evaluated.

Output: 
	A Score of how confident this variant is a de novo variant for each locus.

Procedure:
	1. If Active Region: Assembly three samples together and re-alignment. 
	2. Convert Bam to Tensor. Should be a (3*Depth, weight, height)
	3. Put the Tensor into the Model And Get Score.

Training Data:
	True:
		1. True De novo variants from our previous study.
		2. Artificial De novo. Unrelated individuals, one of them is het, other two is hom ref. 
	False:
		1. Inherited Variants from Trio.
		2. Ramdon position that 3 individual are both hom ref.
		3. Edge cases, should be collect from IGV viewing. Which is hard even for manul view. 

	
Questions:
	1. How to train the model? How to use previous model?
	2. How to evaluate the model? Test set? Sanger confirmation of all evaluated variants?
	3. Most efficent DNN config? We should keep model as simple as possible as long as we don't lose performence. Complete ResNet or Inception designed for Image Recongnation maybe unecessary for our purpose. 
