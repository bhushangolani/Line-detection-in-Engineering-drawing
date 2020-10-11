# Line-detection-in-Engineering-drawing

STEPS FOLLOWED:


--> Read The Image 

--> Cropped The Image Question and Answer ( Split The Image In Two)

--> Perfectly Removing the mesh behind Question Image

--> Applying Thresholding To remove the mesh behind the Answer --(As we are differentiaiting on the basis of intensitities and some answer
								                                                 figures are very light so we get some errors as the mesh cant be removed completely)

--> Auto edge detection and blurring of the image to remove the remaining disturbances.

-->Applied Hough Line Transform to differentiate between solid and dashed lines.

